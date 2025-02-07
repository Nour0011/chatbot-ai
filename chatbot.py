from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

if __name__ == "__main__":
    # Load the pre-trained model and tokenizer from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

    # Set up the initial chat history (this is optional, but can help to keep context in a conversation)
    chat_history_ids = None


    # Function to generate a response based on the user's input
    def chat_with_bot(user_input):
        global chat_history_ids

        # Encode the new user input
        new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")

        # Append the new user input to the chat history
        if chat_history_ids is not None:
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
        else:
            bot_input_ids = new_user_input_ids

        # Generate the response from the model with do_sample set to True
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id,
                                          do_sample=True, temperature=0.7, top_p=0.9, top_k=50)

        # Decode the response and return the output text
        bot_output = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        return bot_output


    # Interactive chat loop
    while True:
        user_input = input("You: ")

        # Exit the chat if the user types 'quit'
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break

        # Get the bot's response
        bot_response = chat_with_bot(user_input)

        # Print the bot's response
        print(f"Bot: {bot_response}")
