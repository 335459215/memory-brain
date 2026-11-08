"""时间感知召回: 从 query 解析时间表达 → 时间窗 → kb_search 后过滤。

背景 (2026-11-08): zvec Query 无 filter 参数, 时间窗只能在融合打分后做后过滤
(与 scene 过滤同一模式: 显式命中不足时回退全量并打标, 不饿死结果)。
memory-brain 的 intention.py 只是提醒不是时间解析, 召回逻辑这里另写。

纯 stdlib, 无第三方依赖。解析失败返回 None (不过滤), 绝不让搜索崩。
"""
import re
import time
from datetime import datetime, timedelta

_WEEKDAY_CN = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
_NUM_CN = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
           "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}

# 完整日期: 2026-11-08 / 2026/11/8 / 2026年11月8日 (可带时间, 忽略)
_RE_FULL_DATE = re.compile(
    r"(?<!\d)(\d{4})\s*[-/年]\s*(\d{1,2})\s*[-/月]\s*(\d{1,2})\s*[日号]?")
# 年月: 2026年11月 / 2026-11
_RE_YM = re.compile(r"(?<!\d)(\d{4})\s*(?:[-/年]\s*(\d{1,2})\s*月|[-/]\s*(\d{1,2}))(?!\s*\d)")
# 月日: 11月8日 / 11月8号
_RE_MD = re.compile(r"(?<!\d)(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]")
# 星期: (上|本|这)?周X / 星期X / 礼拜X
_RE_WEEKDAY = re.compile(r"(上|本|这|上一?)?\s*(?:周|星期|礼拜)\s*([一二三四五六日天])")
# N 单位前: 3天前 / 两周前 / 2个月前 / 半年前 (N 支持中文数字)
_RE_AGO = re.compile(r"(\d+|半|[一二两三四五六七八九十])\s*(天|日|周|星期|个?月|年)\s*前")
# 范围: (最近|过去|近|latest|last)(N)?(单位)
_RE_RANGE = re.compile(
    r"(?:最近|过去|近|latest|last)\s*(\d+|半|[一二两三四五六七八九十])?\s*"
    r"(天|日|周|星期|个?月|年|days?|weeks?|months?|years?)", re.I)


def _ts(dt: datetime) -> int:
    return int(dt.timestamp())


def _day_start(dt: datetime) -> datetime:
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def _monday(dt: datetime) -> datetime:
    return _day_start(dt) - timedelta(days=dt.weekday())


def _month_start(dt: datetime) -> datetime:
    return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def _shift_months(dt: datetime, n: int) -> datetime:
    y, m = dt.year, dt.month + n
    y += (m - 1) // 12
    m = (m - 1) % 12 + 1
    d = min(dt.day, [31, 29 if y % 4 == 0 and (y % 100 != 0 or y % 400 == 0) else 28,
                     31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1])
    return dt.replace(year=y, month=m, day=d)


def _window(start: datetime, end: datetime, label: str) -> dict:
    return {"label": label, "start": _ts(start), "end": _ts(end)}


def _num(s) -> int:
    """'3' / '两' / '半' → int。半 由调用方按单位解释。"""
    if s is None:
        return 0
    if s == "半":
        return -1  # 半个单位, 调用方特判
    if s.isdigit():
        return int(s)
    return _NUM_CN.get(s, 0)


def _unit_days(unit: str) -> int:
    u = unit.lower().lstrip("个")
    if u.startswith(("d", "天", "日")):
        return 1
    if u.startswith(("w", "周", "星期")):
        return 7
    if u.startswith(("m", "月")) and not u.startswith("mi"):
        return 30
    return 365


def _named(q: str, now: datetime):
    """命名时间表达 → 窗口。命中返回 dict, 否则 None。"""
    today0 = _day_start(now)
    if re.search(r"大前天", q):
        return _window(today0 - timedelta(days=3), today0 - timedelta(days=2), "大前天")
    if re.search(r"前天", q):
        return _window(today0 - timedelta(days=2), today0 - timedelta(days=1), "前天")
    if re.search(r"昨天|昨日|yesterday", q, re.I):
        return _window(today0 - timedelta(days=1), today0, "昨天")
    if re.search(r"今天|今日|today", q, re.I):
        return _window(today0, now, "今天")
    if re.search(r"上周|上一周|上礼拜|last\s*week", q, re.I):
        mon = _monday(now) - timedelta(days=7)
        return _window(mon, mon + timedelta(days=7), "上周")
    if re.search(r"本周|这周|这个周|这一周|this\s*week", q):
        return _window(_monday(now), now, "本周")
    if re.search(r"上个月|上月|last\s*month", q, re.I):
        ms = _month_start(_shift_months(now, -1))
        return _window(ms, _month_start(now), "上个月")
    if re.search(r"这个月|本月|这月|this\s*month", q):
        return _window(_month_start(now), now, "这个月")
    if re.search(r"去年|last\s*year", q, re.I):
        ys = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        return _window(ys.replace(year=ys.year - 1), ys, "去年")
    if re.search(r"今年|年初|this\s*year", q):
        return _window(now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0),
                       now, "今年")
    if re.search(r"最近|近期|recently", q):
        return _window(now - timedelta(days=7), now, "最近7天")
    return None


def parse_time_window(query: str, now_ts: float = None, absolute: bool = True):
    """从 query 解析时间窗。返回 {"label","start","end"} 或 None (无时间表达/解析失败)。

    absolute=False 时只识别相对时间表达 (今天/昨天/上周/最近N天/N天前...),
    不识别绝对日期 (2026-08-25 / 11月8日 / 周X) — 2026-11-08 实测: 文档标题
    大量嵌绝对日期 ("用户工作偏好(2026-08-25...)") 会被误判成时间意图, 把文档
    本身过滤掉 (self MRR 0.544→0.399)。自动识别走 absolute=False; 绝对日期
    由调用方显式指定 (time_filter 参数) 时才生效。
    """
    if not query:
        return None
    try:
        now = datetime.fromtimestamp(now_ts if now_ts else time.time())
    except Exception:
        return None
    q = str(query).strip()
    if not q:
        return None

    # 1) 完整日期 2026-11-08 (仅显式模式)
    if absolute:
        m = _RE_FULL_DATE.search(q)
        if m:
            try:
                day = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                return _window(day, day + timedelta(days=1), f"{day:%Y-%m-%d}")
            except Exception:
                pass

        # 2) 年月 2026年11月 / 2026-11
        m = _RE_YM.search(q)
        if m:
            try:
                y = int(m.group(1))
                mo = int(m.group(2) or m.group(3))
                ms = datetime(y, mo, 1)
                return _window(ms, _shift_months(ms, 1), f"{y}年{mo}月")
            except Exception:
                pass

        # 3) 月日 11月8日 (本年; 未来日期则理解为去年)
        m = _RE_MD.search(q)
        if m:
            try:
                day = datetime(now.year, int(m.group(1)), int(m.group(2)))
                if day > now + timedelta(days=1):
                    day = day.replace(year=now.year - 1)
                return _window(day, day + timedelta(days=1), f"{day:%m-%d}")
            except Exception:
                pass

        # 4) 星期 (上|本|这)?周X
        m = _RE_WEEKDAY.search(q)
        if m and m.group(2) in _WEEKDAY_CN:
            try:
                wd = _WEEKDAY_CN[m.group(2)]
                mon = _monday(now)
                if m.group(1) and m.group(1).startswith("上"):
                    mon -= timedelta(days=7)
                day = mon + timedelta(days=wd)
                return _window(day, day + timedelta(days=1), f"{day:%Y-%m-%d}(周{m.group(2)})")
            except Exception:
                pass

    # 5) N 单位前
    m = _RE_AGO.search(q)
    if m:
        raw, unit = m.group(1), m.group(2)
        n = _num(raw)
        if raw == "半":  # 半个单位: 半年→6个月前那个月, 半个月→15天, 其余→1个单位
            if unit in ("月", "个月"):
                start = _month_start(_shift_months(now, -6))
                return _window(start, _month_start(_shift_months(now, -5)), "半年前")
            if unit in ("周", "星期"):
                return _window(now - timedelta(days=4), now - timedelta(days=3), "半周前")
            if unit in ("天", "日"):
                return _window(now - timedelta(hours=12), now - timedelta(hours=6), "12小时前")
            if unit in ("月", "个月"):
                start = _day_start(now) - timedelta(days=15)
                return _window(start, start + timedelta(days=1), "15天前")
            # 半年 → 6 个月前那个月
            start = _month_start(_shift_months(now, -6))
            return _window(start, _month_start(_shift_months(now, -5)), "半年前")
        if n <= 0:
            n = 1  # 未识别 → 1 个单位 (天粒度下限)
        if unit in ("天", "日"):
            start = _day_start(now) - timedelta(days=n)
            return _window(start, start + timedelta(days=1), f"{n}天前")
        if unit in ("周", "星期"):
            start = _monday(now) - timedelta(days=7 * n)
            return _window(start, start + timedelta(days=7), f"{n}周前")
        if unit in ("月", "个月"):
            start = _month_start(_shift_months(now, -n))
            return _window(start, _month_start(_shift_months(now, -(n - 1))), f"{n}个月前")
        start = now.replace(month=1, day=1) - timedelta(days=365 * n)
        return _window(start, now.replace(month=1, day=1) - timedelta(days=365 * (n - 1)),
                       f"{n}年前")

    # 6) 范围 (最近|过去|近) N 单位; N 缺省 → 天粒度默认值
    m = _RE_RANGE.search(q)
    if m:
        n, unit = _num(m.group(1)), m.group(2)
        days = _unit_days(unit)
        if n <= 0:
            n = 7 if days <= 7 else (30 if days == 30 else 365)
            return _window(now - timedelta(days=n), now, f"最近{n}天")
        if days == 1:
            return _window(_day_start(now) - timedelta(days=n - 1), now, f"最近{n}天")
        return _window(now - timedelta(days=days * n), now, f"最近{n}{unit}")

    # 7) 命名表达
    return _named(q, now)


def parse_explicit(spec: str, now_ts: float = None):
    """解析显式 time 参数: '7d'/'2w'/'3m'/'1y'/'2026-11-08'/'2026-11'/自然语言。None=不过滤。"""
    if not spec or not str(spec).strip():
        return None
    s = str(spec).strip().lower()
    if s in ("all", "any", "none", "*"):
        return None
    m = re.fullmatch(r"(\d+)\s*([dwmy]|天|日|周|星期|个?月|年)", s)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        fake = f"最近{n}天" if unit in ("d", "天", "日") else \
               f"最近{n}周" if unit in ("w", "周", "星期") else \
               f"最近{n}个月" if unit in ("m", "月", "个月") else f"最近{n}年"
        return parse_time_window(fake, now_ts)
    if re.fullmatch(r"\d{4}-\d{1,2}", s):
        return parse_time_window(s + "月", now_ts)
    return parse_time_window(s, now_ts)


if __name__ == "__main__":
    NOW = datetime(2026, 11, 8, 12, 0, 0).timestamp()  # 周日
    cases = [
        "昨天", "前天", "大前天", "今天下午", "本周", "这周", "上周", "上个月", "这个月",
        "去年", "今年", "最近3天", "最近一周", "过去30天", "近两个月", "最近", "近期",
        "3天前", "两周前", "三周前", "十天前", "2个月前", "半年前", "1年前",
        "上周三", "这周五", "星期三",
        "2026-11-08", "2026/11/8", "2026年11月8日", "2026-11-08 16:59", "11月8日", "10月1日",
        "2026-11", "2026年10月",
        "LunaTV 播放页切后台回来不播 root cause", "zvec FTS 冒号崩溃", "", None,
    ]
    ok = 0
    for c in cases:
        w = parse_time_window(c, NOW)
        if w:
            f = datetime.fromtimestamp(w["start"]).strftime("%m-%d %H:%M")
            t = datetime.fromtimestamp(w["end"]).strftime("%m-%d %H:%M")
            print(f"  {str(c)[:28]:<30} → {w['label']:<14} [{f} → {t}]")
            ok += 1
        else:
            print(f"  {str(c)[:28]:<30} → None (不过滤)")
    for spec in ["7d", "2w", "3m", "1y", "2026-11-08", "2026-11", "all", "昨天"]:
        w = parse_explicit(spec, NOW)
        print(f"  explicit {spec:<12} → {w['label'] if w else 'None'}")
    # 断言
    assert parse_time_window("昨天", NOW)["label"] == "昨天"
    assert parse_time_window("LunaTV root cause", NOW) is None
    d3 = _day_start(datetime.fromtimestamp(NOW)) - timedelta(days=3)
    assert parse_time_window("3天前", NOW)["start"] == _ts(d3), "3天前起点"
    w = parse_time_window("上周三", NOW)
    assert datetime.fromtimestamp(w["start"]).weekday() == 2, "上周三应是周三"
    assert parse_time_window("两周前", NOW)["label"] == "2周前"
    assert parse_time_window("近两个月", NOW)["label"].startswith("最近2"), "近两个月"
    w = parse_time_window("半年前", NOW)
    assert datetime.fromtimestamp(w["start"]).month == 5, "半年前 ≈ 5月"
    assert parse_time_window("2026-11", NOW)["label"] == "2026年11月"
    assert parse_explicit("7d", NOW)["label"] == "最近7天"
    assert parse_explicit("3m", NOW)["label"].endswith("个月"), "3m 应是月"
    assert parse_explicit("all", NOW) is None
    assert parse_time_window("", NOW) is None and parse_time_window(None, NOW) is None
    # absolute=False (自动识别): 相对表达认, 绝对日期/星期不认 (防标题嵌日期误判)
    assert parse_time_window("用户工作偏好(2026-08-25 三 tab 改造)", NOW, absolute=False) is None
    assert parse_time_window("网络环境(2026-08-09): 用户在国内", NOW, absolute=False) is None
    assert parse_time_window("zvec 0.8 升级 FTS 消毒 白名单 今天", NOW, absolute=False)["label"] == "今天"
    assert parse_time_window("上周修的 CDP 泄漏", NOW, absolute=False)["label"] == "上周"
    assert parse_time_window("3天前", NOW, absolute=False)["label"] == "3天前"
    assert parse_time_window("星期三", NOW, absolute=False) is None
    # 绝对日期仍可显式解析
    assert parse_time_window("2026-11-08", NOW)["label"] == "2026-11-08"
    print(f"\n自测通过: {ok}/{len(cases)} 命中, 断言全过")
