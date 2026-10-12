# -*- coding: utf-8 -*-
"""输入侧脱敏：PII 字段替换 + 克罗恩相关整句删除 + IBD 专用药物整句删除。"""

from __future__ import annotations

import re

PII_PLACEHOLDER = "[已隐去]"

# 命中则删除整句（含历史占位符，便于重脱敏）
_CD_SENTENCE_HIT = re.compile(
    r"(?:"
    r"克罗恩|克隆氏?病|克恩病|Crohn'?s?(?:\s*disease)?"
    # OCR 错字变体：克隆恩、克岁恩、克罗病、克罗怘病等
    r"|克[\u4e00-\u9fff]?恩|克[罗隆][\u4e00-\u9fff]?[病氏]"
    r"|\[疾病名已隐去\]"
    r"|(?<![A-Za-z])CD(?![A-Za-z])"
    r")",
    flags=re.I,
)

# IBD 专用药物（氨基水杨酸、生物制剂、免疫抑制剂、肠内营养治疗），命中则删除整句
_DRUG_SENTENCE_HIT = re.compile(
    r"(?:"
    r"美[沙莎]拉[秦嗪琴]|柳氮|SASP|氨基水杨酸|巴柳氮|奥沙拉秦|艾迪莎|颇得[斯思]安|[莎沙砂]尔福|5-?ASA"
    r"|英夫利[西昔]|类克|阿达木|修美乐|乌司奴|喜达诺|优特克|维得利珠|安吉优|生物制剂|IFX|UST|ADA(?![A-Za-z])"
    # 生物类似药与商品名：维多珠单抗、英夫利西单抗类似药（类停）、阿达木单抗类似药等
    r"|维多利?珠|恩塔维奥|类停|安[建健]宁|格乐立|苏立信|汉达远|君迈康|赛乐信|单抗"
    r"|硫唑嘌呤|依木兰|巯嘌呤|6-?MP|甲氨[蝶喋碟]呤|MTX(?![A-Za-z])|沙利度胺|他克莫司|环孢素"
    r"|全肠内营养|EEN|肠内营养治疗"
    r")"
)

# ---- PII ----
_STAFF_LABELS = (
    r"医师签字|主治医师|住院医师|记录者|记录医师|(?:华海历史)?报告医师|审核医师|报告医生|审核医生"
    r"|送检医师|制片校对|检查医师|检查医生|操作医师|操作医生|内镜医师|诊断医师|诊断医生|重核医生"
    r"|护士|麻醉医师|麻醉医生"
)
# 影像报告签名把医生姓名写在标签之前（如「张三 李四审核医师：」或姓名单独成行），仅删除形如姓名的片段
_SIGN_LABELS = r"(?:华海历史)?报告医师|审核医师|报告医生|审核医生"
_NAME_ONE = r"[A-Za-z]?(?:\([0-9A-Za-z]{1,8}\))?"
_NAME_TOKENS = (
    rf"[\u4e00-\u9fff]{{2,3}}{_NAME_ONE}(?:[ \t\u3000]+[\u4e00-\u9fff]{{2,4}}{_NAME_ONE}){{0,2}}[ \t\u3000]*"
)
_STAFF_NAME_PREFIX = re.compile(rf"(?m)^{_NAME_TOKENS}(?=(?:{_SIGN_LABELS})\s*[:：])")
_STAFF_NAME_LINE = re.compile(rf"(?m)^{_NAME_TOKENS}\n(?=(?:{_SIGN_LABELS})\s*[:：])")
# PDF 抽取可能把签名插在正文行中间（「…未见异张三 华海历史报告医师：」），以「姓名+空格+签名标签」识别
_STAFF_NAME_INLINE = re.compile(rf"[\u4e00-\u9fff]{{2,3}}{_NAME_ONE}[ \t\u3000]+(?=(?:{_SIGN_LABELS})\s*[:：])")
# 影像报告联合签名「审核医师：[已隐去] 张三 / 李四注：」
_STAFF_COSIGN = re.compile(
    r"(医师：\[已隐去\])[ \t]*(?:(?:[\u4e00-\u9fff]{2,3}[ \t]*)?/[ \t]*[\u4e00-\u9fff]{2,3}?(?=注|\s|$)"
    r"|[\u4e00-\u9fff]{2,3}(?=注[:：]))"
)
# 内镜报告签名标签的 OCR 错字（「甫核医生：张三报告医生：」「服告医生：张三」）
_STAFF_OCR_LABEL = re.compile(
    r"((?:[甫宙审重]核|[服报扳]告)医[生师])\s*[:：]\s*(?!\[已隐去\])[\u4e00-\u9fff]{2,3}?(?=报告|审核|\s|$)"
)
_REPORT_UUID = re.compile(r"[ \t]*[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
# 外院病理切片编号（「123456-1：结肠粘膜组织」）
_PATH_SLIDE_NO = re.compile(r"(?<![\d\-])\d{5,8}-\d{1,2}(?=[：:])")
# OCR 把「审核医师」「报告医师」拆开，姓名夹在中间（「审核医 张三 李四师: 师:」）
_SPLIT_NAME = r"[\u4e00-\u9fff]{2,3}?(?:\(\d+\))?(?=[ \t]|师|[:：]|$)"
_STAFF_SPLIT_LABEL = re.compile(rf"(?m)(审核医|报告医)[ \t]+{_SPLIT_NAME}(?:[ \t]+(?!报告医){_SPLIT_NAME})?")
# 内镜截图 OCR 把姓名单独放在签名/时间行或已脱敏行之后（「审核时间：…\n张三」「重核医生：\n张三」）
_STAFF_NAME_AFTER_SIGN = re.compile(
    r"(?m)^([^\n]*(?:审核|重核|报告医|报告期|日期|时间|\[已隐去\])[^\n]*\n)[\u4e00-\u9fff]{2,3}[ \t]*(?=\n[ \t]*\n|\n\d|\s*\Z)"
)
_RECORD_NUMBER = re.compile(
    r"(住院号|病案号|病人号|登记号|门诊号|检查编号|放射编号|检查号|影像号|病理号|申请单号|条码号)"
    r"\s*[:：]?\s*(?!\[已隐去\])[A-Za-z]?\d[\dA-Za-z\-]{3,}"
)
_STAFF_VALUE = re.compile(rf"({_STAFF_LABELS})(?!\s*[:：]?\s*\[已隐去\])\s*[:：]?\s*\S{{0,12}}")

# 内镜系统截图 OCR 的抬头块（「ES … 报告已出 … 收起详情/申请单」）只含医生姓名、患者 ID、检查编号与时间，整块删除
_ENDO_SCREENSHOT_HEADER = re.compile(r"(?m)^(?:历史\s*\n)?ES\s*\n[\s\S]{0,400}?(?:收起详情|申请单)[^\n]*\n?")
_ENDO_REPORTED_BY = re.compile(r"[\u4e00-\u9fff]{2,4}(?=报告已出)")
# 院内患者 ID（10 位，以 0000 开头）
_HOSPITAL_PID = re.compile(r"(?<!\d)0{3,4}\d{5,7}(?!\d)")
# 截图抬头残留的检查/登记编号（单独成行的「A1234567890」「E012345」）
_BARE_CODE_LINE = re.compile(r"(?m)^[A-Z]{1,3}\d{5,}[ \t]*$")

_ID18 = re.compile(
    r"(?<!\d)[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)"
)
_ID15 = re.compile(r"(?<!\d)[1-9]\d{7}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}(?!\d)")

_LABELED_FIELDS = [
    (re.compile(r"(身份证号?|证件号码?|居民身份证)\s*[:：]?\s*[0-9Xx＊*]{10,20}"), r"\1：" + PII_PLACEHOLDER),
    (re.compile(r"(姓名|患者姓名|病人姓名)\s*[:：]\s*[^\s，,；;。:\n]{1,8}"), r"\1：" + PII_PLACEHOLDER),
    (re.compile(r"(现住址|住址|地址|联系地址)\s*[:：]\s*[^\n]{2,80}"), r"\1：" + PII_PLACEHOLDER),
    (re.compile(r"(邮编|邮政编码)\s*[:：]?\s*\d{6}"), r"\1：" + PII_PLACEHOLDER),
    (re.compile(r"(电话|手机|联系电话|联系方式|家属电话)\s*[:：]?\s*[\d\-－—\s]{7,20}"), r"\1：" + PII_PLACEHOLDER),
    (
        re.compile(
            r"(住院号|病案号|病人号|病人ID|病人编号|登记号|门诊号|检查编号|放射编号|检查号|影像号|病理号|申请单号|条码号|accession|Accession)"
            r"\s*[:：]?\s*[A-Za-z0-9\-]{4,}",
            re.I,
        ),
        r"\1：" + PII_PLACEHOLDER,
    ),
    (re.compile(r"(床号)(?!\s*[:：]?\s*\[已隐去\])\s*[:：]?\s*\S{1,10}"), r"\1：" + PII_PLACEHOLDER),
    (re.compile(r"(年龄)\s*[:：]?\s*\d{1,3}\s*岁?"), r"\1：" + PII_PLACEHOLDER),
    (re.compile(r"(性别)\s*[:：]?\s*[男女]"), r"\1：" + PII_PLACEHOLDER),
    # 必须带冒号，避免误伤「××民族医学院」等机构名
    (re.compile(r"(民族)\s*[:：]\s*(?!\[已隐去\])\S{1,8}"), r"\1：" + PII_PLACEHOLDER),
    (re.compile(r"(婚姻|婚姻状况)\s*[:：]?\s*\S{1,8}"), r"\1：" + PII_PLACEHOLDER),
    (re.compile(r"(职业)\s*[:：]?\s*[^\n，,]{1,20}"), r"\1：" + PII_PLACEHOLDER),
    (re.compile(r"(病史叙述者|与患者关系)\s*[:：]?\s*[^\n]{1,20}"), r"\1：" + PII_PLACEHOLDER),
    (
        re.compile(rf"({_STAFF_LABELS})(?!\s*[:：]?\s*\[已隐去\])\s*[:：]?\s*\S{{0,12}}"),
        r"\1：" + PII_PLACEHOLDER,
    ),
]

# 报告抬头的院名行（仅新版肠镜模板有，与检查年份和标签相关）
_HOSPITAL_TITLE_LINE = re.compile(
    r"(?im)^[ \t]*(?:[\u4e00-\u9fff（）()]{2,24}医院[ \t]*\n[ \t]*)?The .{0,60}(?:Hospital|University)[^\n]*\n?"
    r"|^[ \t]*中山大学附属第六医院[^\n]*\n?"
)
# 新版内镜报告的无前缀签名「医生：张三」「助手：李四」
_BARE_STAFF_LABEL = re.compile(
    r"(?m)((?:^|(?<=[\s：:；;。]))(?:医生|医师|助手))\s*[:：]\s*(?!\[已隐去\])[\u4e00-\u9fff]{2,3}?(?=麻醉|报告|审核|护士|内镜|医生|\s|$)"
)
_PHONE_CN = re.compile(r"(?<!\d)(?:\+?86[-\s]?)?1[3-9]\d{9}(?!\d)")
_IP = re.compile(r"(?:IP\s*[:：]?\s*)\d{1,3}(?:\.\d{1,3}){3}", re.I)
_USER_LINE = re.compile(r"User\s*[:：]\s*\S+")


def _compact_spaced_labels(text: str) -> str:
    pairs = [
        (r"姓\s*名", "姓名"),
        (r"性\s*别", "性别"),
        (r"年\s*龄", "年龄"),
        (r"住\s*院\s*号", "住院号"),
        (r"门\s*诊\s*号", "门诊号"),
        (r"影\s*像\s*号", "影像号"),
        (r"检\s*查\s*号", "检查号"),
        (r"病\s*案\s*号", "病案号"),
        (r"床\s*号", "床号"),
        (r"身\s*份\s*证\s*号?", "身份证号"),
        (r"现\s*住\s*址", "现住址"),
        (r"联\s*系\s*电\s*话", "联系电话"),
        (r"审\s*核\s*医\s*师", "审核医师"),
        (r"报\s*告\s*医\s*师", "报告医师"),
    ]
    t = text
    for pat, rep in pairs:
        # OCR 常在字间插入间隔点（「姓‧名」）
        t = re.sub(pat.replace(r"\s*", r"[\s‧·•・]*"), rep, t)
    return t


def mask_pii(text: str) -> str:
    if not text:
        return ""
    t = _compact_spaced_labels(text)
    t = _ENDO_SCREENSHOT_HEADER.sub("", t)
    t = _ENDO_REPORTED_BY.sub("", t)
    t = _HOSPITAL_PID.sub(PII_PLACEHOLDER, t)
    t = _BARE_CODE_LINE.sub(PII_PLACEHOLDER, t)
    t = _ID18.sub(PII_PLACEHOLDER, t)
    t = _ID15.sub(PII_PLACEHOLDER, t)
    t = _STAFF_NAME_LINE.sub("", t)
    t = _STAFF_NAME_PREFIX.sub("", t)
    t = _STAFF_NAME_INLINE.sub("", t)
    for pat, repl in _LABELED_FIELDS:
        t = pat.sub(repl, t)
    t = _PHONE_CN.sub(PII_PLACEHOLDER, t)
    t = _HOSPITAL_TITLE_LINE.sub("", t)
    t = _BARE_STAFF_LABEL.sub(r"\1：" + PII_PLACEHOLDER, t)
    t = _IP.sub("IP：" + PII_PLACEHOLDER, t)
    t = _USER_LINE.sub("User：" + PII_PLACEHOLDER, t)
    t = re.sub(r"[＊*]{4,}", PII_PLACEHOLDER, t)
    return t


_SENT_BREAK = "\x00"
# 报告骨架（报告头、字段标签、签名）后面常无句末标点，不设句界会被并入下一句病名描述一起删掉，
# 删除痕迹（缺报告头、缺诊断意见）本身即成为标签线索。
_STRUCT_LABELS = r"(?:诊断意见|影像表现|审核医|报告医|审核日期|审核时间|报告日期|审核期)"
_STRUCT_BEFORE = re.compile(rf"(?=(?:MR|CT) 诊断报告书|{_STRUCT_LABELS}|注：此报告|【内镜|【影像诊断)")
# 科室名含病名（专病门诊/病区）时只隐去科室值，不删整个报告头
_DEPT_WITH_CD = re.compile(r"((?:科\s*室|科别|申请科室)\s*[:：]\s*)[^\s:：]*?(?:克罗恩|克隆氏?病|克恩病)[^\s:：]*")
_ORPHAN_CLOSE_QUOTE = re.compile(r"^(\s*)”")
_STRUCT_AFTER = re.compile(r"((?:影像表现|诊断意见|检查所见|内镜所见|内镜诊断|影像诊断)\s*[:：】]?)")


def _iter_sentences(text: str):
    """按句末标点与报告结构标签分句；普通换行不当句界，避免『结果示：\\n…克罗恩…』被拆开后留下空头。"""
    buf: list[str] = []
    for ch in text:
        if ch == _SENT_BREAK:
            if buf:
                yield "".join(buf)
            buf = []
            continue
        buf.append(ch)
        if ch in "。！？；":
            yield "".join(buf)
            buf = []
    if buf:
        yield "".join(buf)


def _normalize_wrapped_disease_names(text: str) -> str:
    """合并被换行/空格拆开的病名，避免『克\\n罗恩』漏检。"""
    t = text
    t = re.sub(r"克\s*罗\s*恩", "克罗恩", t)
    t = re.sub(r"克\s*隆\s*氏?\s*病", "克隆病", t)
    t = re.sub(r"克\s*恩\s*病", "克恩病", t)
    t = re.sub(r"C\s*r\s*o\s*h\s*n\s*'?\s*s?", "Crohn", t, flags=re.I)
    # 中文之间的硬换行（版面折行）收拢为一句；报告结构标签保持独占一行
    t = re.sub(rf"([\u4e00-\u9fff])\n+(?!{_STRUCT_LABELS})([\u4e00-\u9fff])", r"\1\2", t)
    return t


def mask_cd_mentions(text: str, drop_drugs: bool = True, report: bool = False) -> str:
    """含克罗恩/CD（或旧版病名占位符）的整句删除，避免削词造成残缺句；默认同时删除含 IBD 专用药物的句子。

    report=True 时按报告骨架设句界；叙述段不设，否则会留下悬空的「内镜诊断：」。
    """
    if not text:
        return ""
    text = _normalize_wrapped_disease_names(text)
    text = _DEPT_WITH_CD.sub(r"\1" + PII_PLACEHOLDER, text)
    if report:
        text = _STRUCT_BEFORE.sub(_SENT_BREAK, text)
        text = _STRUCT_AFTER.sub(r"\1" + _SENT_BREAK, text)
    kept: list[str] = []
    open_quote = dropped_prev = False
    for sent in _iter_sentences(text):
        if _CD_SENTENCE_HIT.search(sent) or (drop_drugs and _DRUG_SENTENCE_HIT.search(sent)):
            open_quote = sent.rfind("“") > sent.rfind("”")
            dropped_prev = True
            continue
        if dropped_prev and sent.lstrip().startswith("”") and kept:
            kept[-1] = re.sub(r"[；;，,、]\s*$", "", kept[-1])
        dropped_prev = False
        if open_quote:
            # 被删句留下未闭合的左引号时，引号内剩余片段一并删除，否则孤立的右引号会暴露删除位置
            close, opening = sent.find("”"), sent.find("“")
            if close >= 0 and (opening < 0 or close < opening):
                sent = _ORPHAN_CLOSE_QUOTE.sub(r"\1", sent[close:])
                sent = re.sub(r"^(\s*)[。，、；]", r"\1", sent)
            elif close < 0 and len(sent) <= 40 and "\n" not in sent.strip():
                continue
        open_quote = False
        kept.append(sent)
    t = "".join(kept)
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[。]{2,}", "。", t)
    return t.strip()


def _mask_staff_and_numbers(text: str) -> str:
    """整句删除会合并中文硬换行，可能把单独成行的医生姓名与签名标签重新拼到一起，需再清理一遍。"""
    t = _STAFF_NAME_LINE.sub("", text)
    t = _STAFF_NAME_PREFIX.sub("", t)
    t = _STAFF_NAME_INLINE.sub("", t)
    t = _STAFF_VALUE.sub(r"\1：" + PII_PLACEHOLDER, t)
    t = _STAFF_NAME_AFTER_SIGN.sub(r"\1" + PII_PLACEHOLDER, t)
    t = _STAFF_COSIGN.sub(r"\1", t)
    t = _STAFF_OCR_LABEL.sub(r"\1：" + PII_PLACEHOLDER, t)
    t = _REPORT_UUID.sub("", t)
    t = _PATH_SLIDE_NO.sub(PII_PLACEHOLDER, t)
    t = _STAFF_SPLIT_LABEL.sub(r"\1 " + PII_PLACEHOLDER, t)
    t = _STAFF_TITLE_NARRATIVE.sub(PII_PLACEHOLDER, t)
    t = _EXT_PATH_NO.sub(PII_PLACEHOLDER, t)
    return _RECORD_NUMBER.sub(r"\1：" + PII_PLACEHOLDER, t)


# 叙述段落首尾的残句（整句删除后留下的「”收住我科。」「结合患者病史及」）与标签相关
_NARR_LEAD_FRAGMENT = re.compile(r"^[”\"'，,、。；;：:）)\s]+[^。\n]{0,40}?。\s*|^[”\"'，,、。；;：:）)\s]+")
_NARR_TAIL_FRAGMENT = re.compile(r"(?<=[。！？])[^。！？]{1,60}$")


def tidy_narrative(text: str) -> str:
    t = _NARR_LEAD_FRAGMENT.sub("", text.strip())
    t = re.sub(rf"^{_LIST_NO}", "", t)
    if t and not re.search(r"[。！？.）)]$", t):
        t = _NARR_TAIL_FRAGMENT.sub("", t)
    return t.strip()


_STAFF_TITLE_NARRATIVE = re.compile(
    r"[\u4e00-\u9fff]{2,3}?(?=(?:副?主任医师|主治医师|副?教授|主任)(?:查房|指示|示|认为|意见))"
)
_EXT_PATH_NO = re.compile(r"[ⅠⅡⅢ]\s*\d{2}-\d{3,}(?:-\d+)?")


# 整句删除会让编号列表出现缺口（如诊断意见从「2.」开始），缺口本身与标签相关，故统一去掉列表序号
_LIST_NO = r"(?:\d{1,2}[ \t]*[.、．]|[（(]?\d{1,2}[）)])[ \t]*(?!\d)"
_LIST_NO_LINE = re.compile(rf"(?m)^([ \t]*){_LIST_NO}")
# 去掉序号后条目行以汉字开头，会被中文硬换行合并规则并入上一条；先给上一条补句号
_LIST_PREV_UNTERMINATED = re.compile(rf"([\u4e00-\u9fff])((?:\n[ \t]*)+)(?={_LIST_NO})")
_LIST_NO_INLINE = re.compile(r"(?<=[。；;：:])[ \t]*(?:\d{1,2}[ \t]*[.、．]|[（(]\d{1,2}[）)])[ \t]*(?!\d)")
# OCR 错位的病理条目序号「2（. 升结肠）」
_LIST_NO_OCR = re.compile(r"(?<![\d.])\d{1,2}[（(][.．][ \t]*")


def _strip_list_numbers(text: str) -> str:
    t = _LIST_NO_OCR.sub("（", text)
    t = _LIST_PREV_UNTERMINATED.sub(r"\1。\2", t)
    t = _LIST_NO_LINE.sub(r"\1", t)
    return _LIST_NO_INLINE.sub("", t)


def mask_for_model(text: str, report: bool = False) -> str:
    """完整脱敏：先 PII，再克罗恩相关及 IBD 专用药物整句删除，去掉列表序号，最后复查医生姓名与编号。"""
    return _mask_staff_and_numbers(_strip_list_numbers(mask_cd_mentions(mask_pii(text), report=report)))
