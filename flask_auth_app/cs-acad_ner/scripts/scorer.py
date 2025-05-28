import spacy

# Load your trained span-based NER model
nlp = spacy.load("models/cs-acad_spancat1")  # Update path as needed

# Sample academic input text
text = "ENHANCING VIRTUAL CONFERENCING: THE INTEGRATION OF GAMIFICATION IN AN INTERACTIVE 3D SPACE ENVIRONMENT\nRAFAELLA R. BANEZ AALIHYA M. RIVERO RYAN CHRISTIAN M. ROBLES January 2025"

# Apply model
doc = nlp(text)

# Print predicted spans
print("📄 Input Text:")
print(text)
print("\n🔍 Predicted Spans:")
found = False
for span in doc.spans.get("sc", []):  # Use correct span key
    print(f" - Label: {span.label_}, Text: '{span.text}'")
    found = True

if not found:
    print("⚠️  No spans predicted by the model.")
