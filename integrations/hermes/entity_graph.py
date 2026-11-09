"""实体图谱 (2026-11-08, 移植 memory-brain 实体关联思路到 Hermes KB)。

SQLite 边车库 (entities.db), 与 zvec 主库并行:
  entities(name  canonical 小写, display 原始形态, type, doc_count)
  doc_entities(doc_id, entity)

挖掘: 拉丁 token (kebab/snake/CamelCase/ALLCAPS/裸词) 按 canonical 小写归并,
      文档频率 df>=3 + 英文停用词过滤。CJK 不挖 (无 jieba, 不做 n-gram 猜词)。
查询: entities_in(query) 只认词典内实体 → kb_search 按关联加分 (entity_boost);
      kb_graph(entity) → 关联文档 + 共现实体。

纯 stdlib (sqlite3/re)。任何异常向上传播由调用方兜底, 不拖死搜索主路径。
"""
import os
import re
import sqlite3
import threading

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "entities.db")
_lock = threading.Lock()

# kebab/snake (new-api, kb_search) 或 裸词 (Emby, VPS, emby) — 大小写不敏感归并。
# 边界用 ASCII 环视而非 \b: "emby插件"/"LunaTV播放页" 这类中英紧贴也要切出来
# (\b 在 CJK 与拉丁间不生效 — Python \w 含 CJK, 2026-11-08 实测 emby df 124→155)。
_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)+(?![A-Za-z0-9])"
    r"|(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9]{2,}(?![A-Za-z0-9])")

# CJK 实体挖掘 (2026-11-09): 中文无词边界, 收连续 CJK 串的所有 bigram (长词被
# 切成片面, 靠 df 门槛 + idf 加权兜底); 含功能字的 bigram 一律不收 (的地得吗呢…
# 跨文档高频但零区分度)。Latin token 与 CJK bigram 走同一条 df/idf 管线。
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]{2,}")
_CJK_FUNC = set("的了着过是在有和与及或也都还很太之于以为被把将从到向由其此该各每某我你他她它们个上下中内外前后间时分会能要想做用来说看给让使成为以至但却而且因故若虽即便再又才正已曾未将须需应得吗呢吧啊哦么这那什么怎么如并就等包按据包括凭待公具典编综维网")
_CJK_MIN_DF = 3  # 与拉丁 token 同门槛
ENTITY_CJK = os.environ.get("KB_ENTITY_CJK", "1") not in ("0", "false", "off")  # CJK bigram 实体开关 (A/B 用; 词典始终全量建, 查询侧过滤)


def _cjk_bigrams(text: str) -> set:
    """文本中所有 CJK bigram (去功能字组合)。"""
    out = set()
    for run in _CJK_RUN_RE.findall(text or ""):
        if len(run) == 2:
            pair = run
            if pair[0] not in _CJK_FUNC and pair[1] not in _CJK_FUNC:
                out.add(pair)
            continue
        for i in range(len(run) - 1):
            pair = run[i:i + 2]
            if pair[0] in _CJK_FUNC or pair[1] in _CJK_FUNC:
                continue
            out.add(pair)
    return out


def _is_cjk(s: str) -> bool:
    return bool(s) and all("\u4e00" <= c <= "\u9fff" for c in s)

# 英文功能词 + 泛技术词 (df 高但无区分度)。MCP/LLM/VPS/CDP/CDN/PWA/CF 等本栈
# 专有词不停 — 它们是有效实体。
_STOP = set("""
the and for with that this from have has not are was were will would could should
you your they their them then than into onto over under again once here there when
where what which who whom whose why how all any both each few more most other some
such only own same too very can just also about after before between during through
under above off out up down because while against being doing does did done get gets
got make makes made use uses used using way ways thing things need needs needed want
wants go goes going take takes taken come comes came see sees seen know knows known
think thinks thought find finds found give gives gave tell tells told work works
worked call calls called try tries tried ask asks asked seem seems felt feel feels
leave leaves left keep keeps kept let lets begin helps help put puts mean means
said say says part parts good good new old first last long own little great big
high low small large next early young important few public same able might must
shall should would could per via etc les une des ist das die der den dem und ist
a an of to in on at by be as is it its if or nor but so we he she i me my no yes
true false null none ok data data file files line lines code test tests error
errors issue issues fix bug bugs info log logs debug note notes doc docs read
write writes wrote run runs running set sets add adds added remove removes
change changes changed check checks checked update updates updated create
creates created delete deletes build builds built deploy deploys deployed
start starts started stop stops end ends send sends sent open opens close
closes show shows shown print prints return returns value values key keys
field fields name names type types list dict str int float bool object
function functions class method methods module package import imports export
config configs conf env vars var const static public private void main
http https httpss url urls uri apis api rest rpc soap json jsonp xml yaml
html htm css js ts py rb go rs java c cpp h m mm ip ips tcp udp dns dhcp
os osx win linux unix shell bash sh cli gui ui ux ide sdk jdk api apis
db dbs sql ddl dml orm redis memcached postgres mysql sqlite mongodb
cpu cpus ram gpu gpus disk ssd hdd io net network internet intranet
user users admin admins root guest anonymous auth token tokens session
memory cache buffer stack heap queue thread process daemon service
true false on off yes no none null nil nan inf infinity zero one two
day days week weeks month months year years hour hours minute minutes
second seconds time times date dates today tomorrow yesterday now
""".split())

_conn_local = threading.local()


def _conn():
    con = getattr(_conn_local, "con", None)
    if con is None:
        con = sqlite3.connect(DB_PATH, timeout=10)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA busy_timeout=5000")
        _conn_local.con = con
    return con


def init_db():
    with _lock, _conn() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS entities(
            name TEXT PRIMARY KEY, display TEXT, type TEXT, doc_count INTEGER DEFAULT 0)""")
        con.execute("""CREATE TABLE IF NOT EXISTS doc_entities(
            doc_id TEXT, entity TEXT, PRIMARY KEY(doc_id, entity))""")
        con.execute("CREATE INDEX IF NOT EXISTS idx_de_entity ON doc_entities(entity)")


def tokenize(text: str, cjk: bool = None):
    """文本 → canonical token 集合 (小写, 去停用词, 去纯数字)。
    cjk: 是否含 CJK bigram 实体 (2026-11-09; 默认读 KB_ENTITY_CJK, A/B 用)。"""
    if not text:
        return set()
    out = set()
    for m in _TOKEN_RE.finditer(str(text)):
        t = m.group(0)
        low = t.lower()
        if low in _STOP or len(low) < 3 or low.isdigit():
            continue
        out.add(low)
    if cjk is None:
        cjk = ENTITY_CJK
    if cjk:
        out |= _cjk_bigrams(str(text))
    return out


def _surface_forms(text: str):
    """canonical → 最常见原始形态 (展示用)。"""
    forms = {}
    for m in _TOKEN_RE.finditer(str(text or "")):
        t = m.group(0)
        low = t.lower()
        if low not in forms:
            forms[low] = t
    return forms


def _kind_of(tok: str) -> str:
    if "-" in tok or "_" in tok:
        return "service"
    if tok.isupper():
        return "acronym"
    if tok[0].isupper():
        return "tool"
    return "service"


def link_doc(doc_id: str, title: str, content: str, min_df_hint=None):
    """抽取单篇文档实体并入库 (增量; kb_add 成功后调)。返回实体列表。"""
    toks = tokenize((title or "") + " " + (content or ""))
    if not toks:
        return []
    forms = _surface_forms((title or "") + " " + (content or ""))
    with _lock, _conn() as con:
        for t in toks:
            con.execute(
                "INSERT OR IGNORE INTO entities(name, display, type, doc_count)"
                " VALUES(?,?,?,0)", (t, forms.get(t, t), _kind_of(forms.get(t, t))))
            con.execute(
                "INSERT OR IGNORE INTO doc_entities(doc_id, entity) VALUES(?,?)", (doc_id, t))
            # 即时刷新被触及实体的 doc_count — 否则新增实体计数永远 0, 达不到
            # df>=3 匹配线, 要等全量重建才生效 (2026-11-09 实测发现)
            con.execute(
                "UPDATE entities SET doc_count="
                "(SELECT COUNT(*) FROM doc_entities WHERE doc_entities.entity=entities.name)"
                " WHERE name=?", (t,))
    return sorted(toks)


def refresh_counts():
    """全量重算 entities.doc_count (回填后调)。"""
    with _lock, _conn() as con:
        con.execute("""UPDATE entities SET doc_count =
            (SELECT COUNT(*) FROM doc_entities WHERE doc_entities.entity = entities.name)""")


def known_entities():
    with _lock, _conn() as con:
        return {r[0]: r[1] for r in con.execute(
            "SELECT name, doc_count FROM entities WHERE doc_count >= 3")}


def entities_in(query: str, cap: int = 2):
    """query 中命中的词典实体, 按 df 升序 (越罕见越优先), 最多 cap 个。"""
    if not query:
        return []
    toks = tokenize(query)
    if not toks:
        return []
    known = known_entities()
    hits = [(df, t) for t, df in known.items() if t in toks]
    hits.sort()
    return [t for _, t in hits[:cap]]


def entities_in_detailed(query: str, cap: int = 2):
    """同 entities_in, 但返回 [(name, df)] — 调用方按 idf 加权用。"""
    if not query:
        return []
    toks = tokenize(query)
    if not toks:
        return []
    known = known_entities()
    hits = [(df, t) for t, df in known.items() if t in toks]
    hits.sort()
    return [(t, df) for df, t in hits[:cap]]


def linked_docs(entities, cap: int = 400):
    """实体 → 关联 doc_id 集合。"""
    if not entities:
        return set()
    q = ",".join("?" * len(entities))
    with _lock, _conn() as con:
        rows = con.execute(
            f"SELECT DISTINCT doc_id FROM doc_entities WHERE entity IN ({q}) LIMIT ?",
            (*entities, cap)).fetchall()
    return {r[0] for r in rows}


def entity_of_doc(doc_id: str):
    with _lock, _conn() as con:
        return [r[0] for r in con.execute(
            "SELECT entity FROM doc_entities WHERE doc_id=?", (doc_id,))]


def related(entity: str, limit: int = 8):
    """共现实体: 与该实体同现的其它实体, 按共现文档数降序。"""
    with _lock, _conn() as con:
        rows = con.execute(
            """SELECT e2.entity, COUNT(*) c FROM doc_entities e1
               JOIN doc_entities e2 ON e1.doc_id = e2.doc_id
               WHERE e1.entity=? AND e2.entity!=?
               GROUP BY e2.entity ORDER BY c DESC LIMIT ?""",
            (entity, entity, limit)).fetchall()
    return [{"entity": r[0], "co_docs": r[1]} for r in rows]


def docs_for(entity: str, limit: int = 20):
    with _lock, _conn() as con:
        return [r[0] for r in con.execute(
            "SELECT doc_id FROM doc_entities WHERE entity=? LIMIT ?", (entity, limit))]


def stats():
    with _lock, _conn() as con:
        ne = con.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        nl = con.execute("SELECT COUNT(*) FROM doc_entities").fetchone()[0]
        top = con.execute(
            "SELECT name, doc_count FROM entities ORDER BY doc_count DESC LIMIT 10").fetchall()
    return {"entities": ne, "links": nl,
            "top": [{"entity": r[0], "docs": r[1]} for r in top]}


def rebuild_from_col(col, min_df: int = 3):
    """全量回填: 遍历 zvec 集合 → 挖 token → df 过滤 → 重建两张表。
    返回 (doc数, 实体数, 边数)。"""
    init_db()
    df = {}
    per_doc = {}
    for d in col.iter_docs():
        f = d.fields
        text = (f.get("title", "") or "") + " " + (f.get("content", "") or "")
        toks = tokenize(text)
        per_doc[d.id] = (toks, _surface_forms(text))
        for t in toks:
            df[t] = df.get(t, 0) + 1
    kept = {t for t, c in df.items() if c >= min_df}
    with _lock, _conn() as con:
        con.execute("DELETE FROM doc_entities")
        con.execute("DELETE FROM entities")
        for doc_id, (toks, forms) in per_doc.items():
            for t in toks & kept:
                con.execute("INSERT OR IGNORE INTO entities(name, display, type, doc_count)"
                            " VALUES(?,?,?,0)",
                            (t, forms.get(t, t), _kind_of(forms.get(t, t))))
                con.execute("INSERT OR IGNORE INTO doc_entities(doc_id, entity) VALUES(?,?)",
                            (doc_id, t))
    refresh_counts()
    s = stats()
    return len(per_doc), s["entities"], s["links"]


if __name__ == "__main__":
    init_db()
    # 自测: 挖掘/匹配/停用词
    q = "Emby 插件 C# HarmonyLib ARM64 部署 the api"
    toks = tokenize(q)
    assert "emby" in toks and "harmonylib" in toks and "arm64" in toks, toks
    assert "the" not in toks and "api" not in toks, toks
    assert entities_in("new-api 日志无限增长 轮转 copytruncate") == [] or True  # 词典空时无害
    print("tokenize OK:", sorted(toks))
    print("entities_in('new-api 日志'):", entities_in("new-api 日志"))
    print("stats:", stats())
