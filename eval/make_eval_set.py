import json, pathlib

PDF = "挖掘机维修案例(样例）.pdf"
SCAN = "scan_maintenance_notice.pdf"

def item(qid, q, typ, src, page, kws, hist=None, abstain=False, ocr=False):
    return {
        "question_id": qid,
        "type": typ,
        "question": q,
        "history": hist or [],
        "expected_source": None if abstain else src,
        "expected_page": None if abstain else page,
        "expected_keywords": kws,
        "should_abstain": abstain,
        "ocr_required": ocr,
        "human_verified": False,
    }

items = []

# ---- 简单事实 ----
items.append(item("q001", "发动机冒蓝烟案例中，最终采取了什么维修方案？", "simple_fact", PDF, 1, ["更换喷油器"]))
items.append(item("q002", "挖机60E怠速抖动大的案例，最后更换了什么零件？", "simple_fact", PDF, 2, ["脚垫"]))
items.append(item("q003", "水温高报警案例的最终维修方案是什么？", "simple_fact", PDF, 3, ["更换总电源"]))
items.append(item("q004", "停车时间长后不好启动的案例，最后是怎么解决的？", "simple_fact", PDF, 4, ["清洗燃油箱"]))
items.append(item("q005", "发动机冒蓝烟的案例中，断缸测试发现几个缸缺缸？", "simple_fact", PDF, 1, ["4", "缺缸"]))
items.append(item("q006", "水温高报警案例中，实测的实际水温是多少度？", "simple_fact", PDF, 3, ["62"]))

# ---- 型号/编号 ----
items.append(item("q007", "机号 FTC003RLAJX103103 对应的是哪个故障案例？", "model_code", PDF, 1, ["冒蓝烟"]))
items.append(item("q008", "机号 FTC003RFVHH102378 出现在哪个故障案例？", "model_code", PDF, 2, ["怠速抖动"]))
items.append(item("q009", "水温高报警案例的机型与机号分别是什么？", "model_code", PDF, 3, ["FTC003RFTHH102373"]))
items.append(item("q010", "工作小时为 413 小时的设备出现了什么问题？", "model_code", PDF, 4, ["不好启动"]))

# ---- 模糊语义 ----
items.append(item("q011", "柴油味重、燃烧不充分并冒蓝烟，可能是什么故障？", "fuzzy_semantic", PDF, 1, ["喷油器"]))
items.append(item("q012", "设备在低转速下明显发抖、转速升高后恢复正常的毛病是什么？", "fuzzy_semantic", PDF, 2, ["怠速抖动"]))
items.append(item("q013", "仪表显示冷却水温报警但实际水温不高，应该怎么判断？", "fuzzy_semantic", PDF, 3, ["假高温"]))
items.append(item("q014", "机器停放一段时间后偶尔打不着火，燃油箱内部可能有什么问题？", "fuzzy_semantic", PDF, 4, ["铁锈", "水"]))

# ---- 多轮指代（成对出现）----
h1 = [{"question": "挖机怠速抖动大的案例排查了哪些可能原因？", "answer": "（上一轮回答，仅用于提供指代上下文）"}]
h2 = [{"question": "水温高报警的案例是怎么一步步排查处理的？", "answer": "（上一轮回答，仅用于提供指代上下文）"}]
items.append(item("q015", "挖机怠速抖动大的案例排查了哪些可能原因？", "multi_turn", PDF, 2, ["发动机固定螺栓"]))
items.append(item("q016", "那最后更换的是什么零件？", "multi_turn", PDF, 2, ["脚垫"], hist=h1))
items.append(item("q017", "水温高报警的案例是怎么一步步排查处理的？", "multi_turn", PDF, 3, ["水温传感器"]))
items.append(item("q018", "它最后更换的配件是什么？", "multi_turn", PDF, 3, ["总电源"], hist=h2))

# ---- 扫描件（OCR 链路）----
items.append(item("q019", "HPU-01 液压泵站首次保养运行多少小时需要更换吸油滤芯？", "scanned", SCAN, 1, ["500"], ocr=True))
items.append(item("q020", "HPU-01 液压泵站多久更换一次液压油？", "scanned", SCAN, 1, ["2000"], ocr=True))
items.append(item("q021", "HPU-01 液压泵站的保修期限是多久？", "scanned", SCAN, 1, ["12"], ocr=True))
items.append(item("q022", "HPU-01 液压泵站的制造商是哪家公司？", "scanned", SCAN, 1, ["成都液压设备有限公司"], ocr=True))
items.append(item("q023", "HPU-01 液压泵站要求的存放环境温度范围是多少？", "scanned", SCAN, 1, ["5", "40"], ocr=True))

# ---- 应拒答 ----
items.append(item("q024", "挖掘机案例中记录的最低油耗是多少升每小时？", "abstain", None, None, [], abstain=True))
items.append(item("q025", "案例中设备的发动机机油使用的是什么品牌？", "abstain", None, None, [], abstain=True))
items.append(item("q026", "HPU-01 液压泵站的额定工作压力是多少兆帕？", "abstain", None, None, [], abstain=True))

path = pathlib.Path("eval/eval_set.json")
path.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("items:", len(items))
from collections import Counter
print(Counter(i["type"] for i in items))
