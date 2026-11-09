"""Corpus helpers."""
from __future__ import annotations

import json
from pathlib import Path

_DEMO = [
    {"id": "d1", "title": "退款时效", "text": "用户申请退款后，款项通常在 3 到 7 个工作日原路退回。若超过 7 个工作日未到账，请联系客服并提供订单号。"},
    {"id": "d2", "title": "订单号查询", "text": "请在「我的订单」页复制订单号，例如 ORD-2024-881。客服会用该编号核对物流与支付流水。"},
    {"id": "d3", "title": "运费说明", "text": "满 99 元免运费。偏远地区可能加收 5 到 15 元。运费退回规则与商品退款一致。"},
    {"id": "d4", "title": "账号安全", "text": "请勿向他人透露验证码。若发现异常登录，立即修改密码并开启二次验证。"},
    {"id": "d5", "title": "发票须知", "text": "电子发票在订单完成后 24 小时内发送至注册邮箱。纸质发票需在下单时勾选并填写抬头。"},
    {"id": "d6", "title": "退货流程", "text": "支持 7 天无理由退货。请保持商品完好，按页面指引填写退货单，快递费首重由平台承担一次。"},
    {"id": "d7", "title": "优惠券叠用", "text": "满减券与会员折扣不可同时使用。运费券可与满减券叠加。活动商品可能排除部分券。"},
    {"id": "d8", "title": "客服工作时间", "text": "在线客服每日 9:00 至 22:00。紧急订单问题可留言，我们将在下一工作时段优先处理。"},
]


def demo_corpus() -> dict[str, str]:
    return {row["id"]: f"{row['title']}。{row['text']}" for row in _DEMO}


def load_corpus(path: str | Path) -> dict[str, str]:
    docs: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            docs[row["id"]] = f"{row.get('title', '')}。{row.get('text', '')}"
    return docs
