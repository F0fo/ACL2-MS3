from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset
import evaluate
import numpy as np
import json

model_path = "google-bert/bert-base-uncased"

INTENTS = [
    "hotel_recommendation",  # 0
    "hotel_search",          # 1
    "hotel_info",            # 2
    "review_query",          # 3
    "comparison",            # 4
    "traveller_preference",  # 5
    "location_query",        # 6
    "visa_query",            # 7
    "rating_filter",         # 8
    "general_question"       # 9
]


tokenizer = None
model = None

id2label = {i: label for i, label in enumerate(INTENTS)}
label2id = {label: i for i, label in enumerate(INTENTS)}

def load_training_data():
# Load training data
    with open("data/training_data.json", "r") as f:
        training_examples = json.load(f)

    print(f"Loaded {len(training_examples)} training examples")
    return training_examples

def tokenize_function(tokenizer, examples):
    return tokenizer(examples["text"], truncation=True, padding=True)

def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def train_intent_model():
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=len(INTENTS),
        id2label=id2label,
        label2id=label2id
    )


    dataset = Dataset.from_list(load_training_data())



    tokenized_dataset =  dataset.map(
        lambda x: tokenize_function(tokenizer, x),
        batched=True,
    )


    split_dataset = tokenized_dataset.train_test_split(test_size=0.15, seed=42)


    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)



    # Training config
    print("Setting up training...")
    training_args = TrainingArguments(
        output_dir="intent_model",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=15,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train
    print("Training the model")
    trainer.train()

    # Save
    print("Saving the model")
    trainer.save_model("intent_model_final")
    tokenizer.save_pretrained("intent_model_final")




def load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        tokenizer = AutoTokenizer.from_pretrained("intent_model_final")
        model = AutoModelForSequenceClassification.from_pretrained("intent_model_final")
        model.eval()


def classify_intent(text: str):
    """
    Classify user intent.

    Returns:
        {
            "intent": str,
            "confidence": float
        }
    """
    load_model()


    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
    )

    outputs = model(**inputs)

    predicted_class = outputs.logits.argmax(dim=1).item()
    confidence = outputs.logits.softmax(dim=1).max().item()

    return {
        "intent": id2label[predicted_class],
        "confidence": confidence,
    }



if __name__ == "__main__":
    # Test
    print("\nTesting the model...")
    print("=" * 60)

    test_queries = [
        "Recommend me a good hotel in Tokyo",
        "Best hotels for business travelers",
        "Hotels with cleanliness rating above 9",
        "Do I need a visa to travel from India to Dubai?",
        "How many hotels do you have?",
        "Show me reviews for this hotel",
        "Compare The Azure Tower and Marina Bay",
        "Where is The Golden Oasis located?",
        "Find hotels in Paris",
        "Tell me about The Royal Compass"
    ]

    for query in test_queries:
        result = classify_intent(query)

        print(f"\nQuery: {query}")
        print(f"  Intent: {result['intent']}")
        print(f"  Confidence: {result['confidence']:.3f}")

    print("\n" + "=" * 60)
    print("Training complete!")
