import pandas as pd

# Sample data (you can extend this list to 10 identical or varied records)
data = [
    {
        "message_type": "EDI_850_PURCHASE_ORDER",
        "payload": {
            "purchase_order": {
                "po_number": "PO-20260510-1001",
                "po_date": "2026-05-10",
                "currency": "CAD"
            },
            "parties": {
                "buyer": {"gln": "1514032003830"},
                "seller": {"gln": "1514250012345"}
            },
            "line_items": [
                {
                    "line_number": 1,
                    "item_identification": {
                        "gtin_14": "22222333334444",
                        "description": "Industrial Sensor"
                    },
                    "quantity_ordered": 10,
                    "unit_of_measure": "EA",
                    "unit_price": 25.50
                }
            ],
            "totals": {
                "total_quantity_ordered": 10
            }
        }
    }
]

# ---- Expand to 10 rows ----
# Here we simply replicate the same record 10 times for demonstration
data = data * 10

# ---- Flatten JSON into table ----
rows = []

for record in data:
    payload = record["payload"]
    line_item = payload["line_items"][0]

    rows.append({
        "po_number": payload["purchase_order"]["po_number"],
        "po_date": payload["purchase_order"]["po_date"],
        "currency": payload["purchase_order"]["currency"],
        "buyer_gln": payload["parties"]["buyer"]["gln"],
        "seller_gln": payload["parties"]["seller"]["gln"],
        "gtin_14": line_item["item_identification"]["gtin_14"],
        "description": line_item["item_identification"]["description"],
        "quantity_ordered": line_item["quantity_ordered"],
        "unit": line_item["unit_of_measure"],
        "unit_price": line_item["unit_price"],
        "total_quantity_ordered": payload["totals"]["total_quantity_ordered"]
    })

# Create DataFrame
df = pd.DataFrame(rows)

# Display table
print(df)

# Optional: save to CSV
df.to_csv("edi_850_table.csv", index=False)