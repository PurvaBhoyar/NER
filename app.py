import gradio as gr
from transformers import pipeline

# Load the NER pipeline
# This model is balanced for speed and accuracy (PER, ORG, LOC, MISC)
ner_pipeline = pipeline("ner", model="dslim/distilbert-NER", aggregation_strategy="simple")

def ner_analysis(text):
    if not text.strip():
        return []
    
    entities = ner_pipeline(text)
    
    # Custom post-processing to include simple Date detection since CoNLL models miss them
    import re
    date_patterns = [
        r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',  # 12-06-2024
        r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?\b' # June 12, 2024
    ]
    
    for pattern in date_patterns:
        for match in re.finditer(pattern, text):
            # Check if this date isn't already covered by a model entity
            is_covered = any(ent['start'] <= match.start() < ent['end'] for ent in entities)
            if not is_covered:
                entities.append({
                    'entity_group': 'DATE',
                    'start': match.start(),
                    'end': match.end(),
                    'word': match.group()
                })
    
    # Sort entities by start index
    entities = sorted(entities, key=lambda x: x['start'])
    
    formatted_entities = []
    last_idx = 0
    
    for entity in entities:
        if entity['start'] > last_idx:
            formatted_entities.append((text[last_idx:entity['start']], None))
        
        # Map labels to readable names
        label = entity['entity_group']
        if label == "PER": label = "Person"
        if label == "ORG": label = "Organization"
        if label == "LOC": label = "Location"
        if label == "MISC": label = "Miscellaneous"
        if label == "DATE": label = "Date"
        
        formatted_entities.append((text[entity['start']:entity['end']], label))
        last_idx = entity['end']
    
    if last_idx < len(text):
        formatted_entities.append((text[last_idx:], None))
        
    return formatted_entities

# Build the Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔍 Professional NER Application")
    gr.Markdown("Detects **People**, **Organizations**, **Locations**, **Dates**, and **Misc** entities.")
    
    with gr.Row():
        input_text = gr.Textbox(
            label="Input Text", 
            placeholder="Type here... e.g., 'Steve Jobs founded Apple in 1976.'",
            lines=4
        )
    
    output_text = gr.HighlightedText(
        label="Recognized Entities",
        combine_adjacent=True,
        show_legend=True,
        color_map={
            "Person": "#ef4444", 
            "Organization": "#3b82f6", 
            "Location": "#22c55e", 
            "Miscellaneous": "#f59e0b",
            "Date": "#8b5cf6"
        }
    )
    
    submit_btn = gr.Button("Analyze Entities", variant="primary")
    
    gr.Examples(
        examples=[
            "Sundar Pichai is the CEO of Google, working at Mountain View since 2015.",
            "Apple Inc. was founded by Steve Jobs and Steve Wozniak on April 1, 1976.",
            "The Olympic Games will be held in Paris in 2024.",
            "Microsoft announced a new AI breakthrough in Seattle last Friday."
        ],
        inputs=input_text
    )
    
    submit_btn.click(fn=ner_analysis, inputs=input_text, outputs=output_text)


if __name__ == "__main__":
    demo.launch()
