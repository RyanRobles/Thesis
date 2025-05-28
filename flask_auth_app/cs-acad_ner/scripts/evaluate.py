import spacy
from spacy.training import Example
import json

def preprocess_text(text):
    """Ensure consistent tokenization"""
    text = text.replace(" - ", "-")  # Handle hyphens
    text = text.replace(" : ", ": ")  # Colons
    return text.strip()

def resolve_overlaps(spans):
    """Resolve overlapping spans by keeping the longest span"""
    if not spans:
        return []
    
    # Convert to uniform format and ensure integers
    processed = []
    for span in spans:
        if isinstance(span, dict):
            try:
                processed.append((int(span["start"]), int(span["end"]), span["label"]))
            except (ValueError, KeyError):
                continue
        else:
            try:
                processed.append((int(span[0]), int(span[1]), span[2]))
            except (ValueError, IndexError):
                continue
    
    # Sort by start position then length (longest first)
    sorted_spans = sorted(processed, key=lambda x: (x[0], -(x[1] - x[0])))
    
    filtered = []
    if not sorted_spans:
        return filtered
        
    prev_start, prev_end, prev_label = sorted_spans[0]
    
    for curr_start, curr_end, curr_label in sorted_spans[1:]:
        if curr_start >= prev_end:
            filtered.append((prev_start, prev_end, prev_label))
            prev_start, prev_end, prev_label = curr_start, curr_end, curr_label
        else:
            curr_len = curr_end - curr_start
            prev_len = prev_end - prev_start
            if curr_len > prev_len:
                prev_start, prev_end, prev_label = curr_start, curr_end, curr_label
    
    filtered.append((prev_start, prev_end, prev_label))
    return filtered

def compute_tp_fp_fn(predicted, gold):
    """Compute true positives, false positives, and false negatives"""
    tp = set(predicted) & set(gold)
    fp = set(predicted) - set(gold)
    fn = set(gold) - set(predicted)
    return tp, fp, fn

def evaluate_span_model(model_path, data_path, spans_key="sc"):
    """
    Evaluate a span-based NER model
    
    Args:
        model_path: Path to trained spaCy model
        data_path: Path to test data in JSON format
        spans_key: Key used for span annotations (default "sc" for SpanCategorizer)
    """
    # 1. Load model
    try:
        nlp = spacy.load(model_path)
    except OSError:
        raise ValueError(f"Model not found at {model_path}. Train first!")
    
    # 2. Verify the model has a span categorizer
    if "spancat" not in nlp.pipe_names:
        raise ValueError("Model must have a SpanCategorizer component!")
    
    # 3. Load and preprocess data
    with open(data_path) as f:
        raw_data = json.load(f)
    
    # 4. Process examples
    examples = []
    skipped = 0
    all_tp = set()
    all_fp = set()
    all_fn = set()

    for item in raw_data:
        text = preprocess_text(item["text"])
        
        # Get and validate spans
        spans = []
        for span in item.get("spans", {}).get(spans_key, item.get("entities", [])):
            if isinstance(span, dict):
                try:
                    spans.append((int(span["start"]), int(span["end"]), span["label"]))
                except (ValueError, KeyError):
                    continue
            else:
                try:
                    spans.append((int(span[0]), int(span[1]), span[2]))
                except (ValueError, IndexError):
                    continue
        
        # Resolve overlaps before creating examples
        spans = resolve_overlaps(spans)
        
        try:
            doc = nlp.make_doc(text)

            # Create aligned spans
            aligned_spans = []
            for start, end, label in spans:
                span = doc.char_span(start, end, label=label)
                if span is not None:
                    aligned_spans.append(span)
                else:
                    print(f"Skipping misaligned span: {text[start:end]} (label: {label})")

            # Create example using aligned spans only
            aligned_span_data = {
                spans_key: [(span.start_char, span.end_char, span.label_) for span in aligned_spans]
            }
            example = Example.from_dict(doc, {"spans": aligned_span_data})
            examples.append(example)

            # Predict using the model
            pred_doc = nlp(text)
            pred_spans = [
                (span.start_char, span.end_char, span.label_)
                for span in pred_doc.spans.get(spans_key, [])
            ]

            gold_spans = [
                (span.start_char, span.end_char, span.label_)
                for span in aligned_spans
            ]

            tp, fp, fn = compute_tp_fp_fn(pred_spans, gold_spans)
            all_tp.update(tp)
            all_fp.update(fp)
            all_fn.update(fn)

        except ValueError as e:
            skipped += 1
            print(f"Skipping: {text[:50]}... - Error: {e}")
    
    print(f"\nProcessed {len(examples)} examples (skipped {skipped})")
    
    # 5. Evaluate
    if not examples:
        raise ValueError("No valid examples to evaluate! Check your data.")
    
    scores = nlp.evaluate(examples)
    print("\nEvaluation Results:")
    print(f"- Span F1: {scores['spans_sc_f']:.3f}")
    print(f"- Span Precision: {scores['spans_sc_p']:.3f}")
    print(f"- Span Recall: {scores['spans_sc_r']:.3f}\n")

    print("Manual Span Evaluation:")
    print(f"- True Positives: {len(all_tp)}")
    print(f"- False Positives: {len(all_fp)}")
    print(f"- False Negatives: {len(all_fn)}")

    if len(all_tp) + len(all_fp) > 0:
        precision = len(all_tp) / (len(all_tp) + len(all_fp))
    else:
        precision = 0.0

    if len(all_tp) + len(all_fn) > 0:
        recall = len(all_tp) / (len(all_tp) + len(all_fn))
    else:
        recall = 0.0

    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    print(f"- Manual Precision: {precision:.3f}")
    print(f"- Manual Recall: {recall:.3f}")
    print(f"- Manual F1 Score: {f1:.3f}")
    
  #  print("Per-Category Performance:")
  #  if 'spans_sc_per_type' in scores:
   #     for label, metrics in scores["spans_sc_per_type"].items():
   #         print(f"{label}:")
  #        print(f"  Precision: {metrics['p']:.3f}")
   #         print(f"  Recall: {metrics['r']:.3f}")
    #        print(f"  F1: {metrics['f']:.3f}\n")
            
  #  else:
     #   print("No per-type metrics available. Check your evaluation data.")

if __name__ == "__main__":
    evaluate_span_model(
        model_path="models/cs-acad_spancat",  # Path to span-based model
        data_path="data/converted_span_data1.json",     # Test data path
        spans_key="sc"                       # Span key used in annotations
    )