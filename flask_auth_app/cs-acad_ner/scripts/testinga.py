import spacy

# Load a blank model (or your actual model if needed)
nlp = spacy.blank("en")  # or "xx" if multilingual

# Sample data
train_data = [
  {
    "text": "CATIPAY, ARISTON L.",
    "spans": {
      "sc": [
        {
          "start": 0,
          "end": 19,
          "label": "AUTHOR"
        }
      ]
    }
  },
  {
    "text": "RAMOS, HARVI M.",
    "spans": {
      "sc": [
        {
          "start": 0,
          "end": 14,
          "label": "AUTHOR"
        }
      ]
    }
  },
  {
    "text": "SILVA, JIESILLE M.",
    "spans": {
      "sc": [
        {
          "start": 0,
          "end": 17,
          "label": "AUTHOR"
        }
      ]
    }
  },
  {
    "text": "ROMMEL M. MACAPOBRE",
    "spans": {
      "sc": [
        {
          "start": 0,
          "end": 18,
          "label": "AUTHOR"
        }
      ]
    }
  },
  {
    "text": "CHARNELLE P. ESTRIBOR",
    "spans": {
      "sc": [
        {
          "start": 0,
          "end": 21,
          "label": "AUTHOR"
        }
      ]
    }
  },
  {
    "text": "KERCHE! C. PALEN",
    "spans": {
      "sc": [
        {
          "start": 0,
          "end": 16,
          "label": "AUTHOR"
        }
      ]
    }
  },
  {
    "text": "ANDRES, JOHN RYAN L",
    "spans": {
      "sc": [
        {
          "start": 0,
          "end": 19,
          "label": "AUTHOR"
        }
      ]
    }
  },
  {
    "text": "BERBASA, SHAUN PRINCE L",
    "spans": {
      "sc": [
        {
          "start": 0,
          "end": 22,
          "label": "AUTHOR"
        }
      ]
    }
  },
  {
    "text": "JOROLAN, VINCE LOUPER R",
    "spans": {
      "sc": [
        {
          "start": 0,
          "end": 22,
          "label": "AUTHOR"
        }
      ]
    }
  },
  {
    "text": "MATHEW AGUSTIN O, BELLA",
    "spans": {
      "sc": [
        {
          "start": 0,
          "end": 23,
          "label": "AUTHOR"
        }
      ]
    }
  }
]


# Check alignment
for example in train_data:
    text = example["text"]
    doc = nlp.make_doc(text)
    print(f"\n🔍 Text:\n{text}")
    print("-" * 40)

    for span in example["spans"]["sc"]:
        start, end, label = span["start"], span["end"], span["label"]
        snippet = text[start:end]

        aligned = doc.char_span(start, end, label=label, alignment_mode="contract")

        if aligned:
            print(f"✅ Aligned: [{label}] '{snippet}' -> Token span: '{aligned.text}'")
        else:
            print(f"❌ Could NOT align: [{label}] '{snippet}' ({start}-{end})")
