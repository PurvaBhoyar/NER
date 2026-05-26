from transformers import pipeline
try:
    ner = pipeline("token-classification", model="arnabdhar/bert-tiny-ontonotes", aggregation_strategy="simple")
    res = ner("On June 12, 2024, Sundar Pichai met Google employees in California.")
    print("Labels found:")
    for r in res:
        print(f"{r['entity_group']}: {r['word']}")
except Exception as e:
    print(f"Error: {e}")
