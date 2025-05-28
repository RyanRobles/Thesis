import spacy
from spacy.training import Example
import json
import random

# --------------------------
# 1. DATA PREPROCESSING FUNCTIONS
# --------------------------
def preprocess_text(text):
    """Normalize text before tokenization"""
    return text.strip()

def resolve_overlapping_spans(spans):
    """Resolve overlapping spans by keeping the longest span"""
    if not spans:
        return []
    
    # Convert all spans to consistent format (start, end, label)
    processed_spans = []
    for span in spans:
        if isinstance(span, dict):
            processed_spans.append((span["start"], span["end"], span["label"]))
        else:  # assume list/tuple format
            processed_spans.append((span[0], span[1], span[2]))
    
    # Sort by start position then length (longest first)
    sorted_spans = sorted(processed_spans, key=lambda x: (x[0], -(x[1] - x[0])))
    
    filtered_spans = []
    if not sorted_spans:
        return filtered_spans
        
    prev_start, prev_end, prev_label = sorted_spans[0]
    
    for curr_start, curr_end, curr_label in sorted_spans[1:]:
        if curr_start >= prev_end:
            filtered_spans.append((prev_start, prev_end, prev_label))
            prev_start, prev_end, prev_label = curr_start, curr_end, curr_label
        else:
            curr_len = curr_end - curr_start
            prev_len = prev_end - prev_start
            if curr_len > prev_len:
                prev_start, prev_end, prev_label = curr_start, curr_end, curr_label
    
    filtered_spans.append((prev_start, prev_end, prev_label))
    return filtered_spans

def fix_misaligned_spans(text, spans):
    """Ensure spans align with token boundaries"""
    nlp = spacy.blank("en")
    doc = nlp.make_doc(text)
    fixed_spans = []
    
    for span in spans:
        if isinstance(span, dict):
            start, end, label = span["start"], span["end"], span["label"]
        else:
            start, end, label = span[0], span[1], span[2]
        
        spacy_span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if spacy_span is None:
            # Try expanding the span if contracting fails
            spacy_span = doc.char_span(start, end, label=label, alignment_mode="expand")
            if spacy_span is None:
                print(f"Could not align span: {text[start:end]} (label: {label})")
                continue
        fixed_spans.append((spacy_span.start_char, spacy_span.end_char, label))
    
    return fixed_spans

def auto_tag_title(text):
    first_line = text.strip().split("\n")[0]
    start = text.find(first_line)
    end = start + len(first_line)
    return [{"start": start, "end": end, "label": "TITLE"}]

def load_data(filepath):
    """Load and preprocess training data in span format"""
    nlp = spacy.blank("en")
    with open(filepath) as f:
        raw_data = json.load(f)
    
    processed_data = []
    for item in raw_data:
        text = preprocess_text(item["text"])
        
        # Get existing spans from the "sc" key
        spans = item.get("spans", {}).get("sc", [])
        
        # Auto-tag the title span (first line)
        auto_title_span = auto_tag_title(text)
        
        # Combine existing spans and auto-tag title span
        # But avoid duplicates: only add if not already present
        existing_title_spans = [s for s in spans if s["label"] == "TITLE"]
        if not existing_title_spans:
            spans += auto_title_span
        
        # Validate and fix spans
        valid_spans = []
        for span in spans:
            # Validate span boundaries
            if (span["start"] >= span["end"] or 
                span["start"] < 0 or 
                span["end"] > len(text)):
                print(f"Invalid span skipped: {span}")
                continue
            
            # Try to align with token boundaries
            doc = nlp.make_doc(text)
            spacy_span = doc.char_span(
                span["start"], 
                span["end"], 
                label=span["label"],
                alignment_mode="contract"
            )
            
            if spacy_span:
                valid_spans.append({
                    "start": spacy_span.start_char,
                    "end": spacy_span.end_char,
                    "label": span["label"]
                })
            else:
                print(f"Could not align span: {text[span['start']:span['end']]}")
        
        if valid_spans:
            processed_data.append((text, {"spans": {"sc": valid_spans}}))
    
    return processed_data


# --------------------------
# 2. MODEL CONFIGURATION (UPDATED)
# --------------------------

def create_blank_model():
    """Create a blank spacy model with SpanCategorizer pipeline"""
    nlp = spacy.blank("en")

    config = {
        "threshold": 0.5,
        "suggester": {"@misc": "spacy.ngram_suggester.v1", "sizes": [1, 2, 3, 4, 5, 6, 7]}
    }

    if "spancat" not in nlp.pipe_names:
        spancat = nlp.add_pipe("spancat", config=config)

    return nlp, spancat

    

# --------------------------
# 3. TRAINING LOGIC (UPDATED)
# --------------------------

def train_span_based_model(nlp, train_data, output_dir):
    examples = []

    for text, annotations in train_data:
        doc = nlp.make_doc(text)
        aligned_spans = []

        # Extract the list of span annotations
        raw_spans = annotations.get("spans", {}).get("sc", [])

        for span in raw_spans:
            start = span["start"]
            end = span["end"]
            label = span["label"]

            # Try to align to token boundaries
            span_obj = doc.char_span(start, end, label=label, alignment_mode="contract")

            if span_obj is not None:
                aligned_spans.append(span_obj)
            else:
                print(f"⚠️ Could not align span: '{text[start:end]}' ({start}-{end})")

        if not aligned_spans:
            print(f"⚠️ No valid spans in: {text[:60]}... Skipping.")
            continue

        # Create a SpanGroup called "sc"
        doc.spans["sc"] = aligned_spans

        try:
            example = Example.from_dict(doc, {"spans": {"sc": [(span.start_char, span.end_char, span.label_) for span in aligned_spans]}})
            examples.append(example)
        except Exception as e:
            print(f"❌ Error creating example: {e}")

    if not examples:
        raise ValueError("No valid training examples found.")

    # Initialize with examples
    nlp.initialize(lambda: examples[:10])

    # Train
    with nlp.select_pipes(enable="spancat"):
        optimizer = nlp.begin_training()

        for epoch in range(20):
            random.shuffle(examples)
            losses = {}

            for batch in spacy.util.minibatch(examples, size=8):
                nlp.update(batch, drop=0.3, losses=losses, sgd=optimizer)

            print(f"Epoch {epoch}: Loss: {losses.get('spancat', 0):.2f}")

    # Save model
    nlp.to_disk(output_dir)
    print(f"Model saved to {output_dir}")


# --------------------------
# MAIN EXECUTION
# --------------------------

if __name__ == "__main__":
    # Load and preprocess data
    train_data = load_data("data/sample_data.json")
    print(f"Loaded {len(train_data)} training examples")

    # Extract unique labels from training data
    labels = set()
    for _, annots in train_data:
        for span in annots["spans"]["sc"]:
            labels.add(span["label"])
    ENTITY_LABELS = sorted(list(labels))
    print(f"Extracted labels: {ENTITY_LABELS}") 
    
    # Verify some samples
    for text, annots in train_data[:3]:
        print(f"Text: {text[:50]}...")
        print(f"Spans: {annots['spans']['sc']}")    
        print("---")
    
    # Create a blank model and get spancat
    nlp, spancat = create_blank_model()

    # Add labels based on the data
    for label in ENTITY_LABELS:
        spancat.add_label(label)

    # Collect all labels from data to verify
    all_data_labels = set()
    for _, annots in train_data:
        for span in annots['spans']['sc']:
            if isinstance(span, dict):
                all_data_labels.add(span["label"])
            elif isinstance(span, (tuple, list)):
                all_data_labels.add(span[2])

    print("\nLABELS IN DATA:", all_data_labels)
    print("Labels count:", len(spancat.labels))
    print("LABELS IN MODEL:", spancat.labels)

    # Check for missing labels
    missing_labels = all_data_labels - set(spancat.labels)
    if missing_labels:
        print("🚨 MISSING LABELS:", missing_labels)

    # Train the span-based model
    train_span_based_model(nlp, train_data, "models/cs-acad_spancat1")

