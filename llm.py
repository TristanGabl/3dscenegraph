from transformers import BertForQuestionAnswering, BertTokenizer
import torch
import json

# Load the pre-trained model and tokenizer
model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

def get_answer(question, context):
    inputs = tokenizer(question, context, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs.input_ids[0][answer_start:answer_end]))
    return answer

# Example usage
context = """
The objects in the scene are as follows: An apple is on the table. A banana is next to the apple. The table supports both the apple and the banana.
"""
question = "What are possible relationships between these objects? Give a list of relationships."
answer = get_answer(question, context)
print(answer)