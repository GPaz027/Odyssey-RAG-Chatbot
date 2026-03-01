from dotenv import load_dotenv
load_dotenv()
from src.chains.book_chain import TextChain


def main():
    print("=" * 60)
    print("Python Documentation Chatbot - Requests Library")
    print("Type 'quit' to exit")
    print("=" * 60)


    qa = TextChain()

    while True:
        user_query = input("\n You: ").strip()

        if user_query.lower() in ['quit', 'exit', 'q']:
            print("\n Goodbye!")
            break

        response = qa.rag_chain.invoke(user_query)
        
        if response:
            print(response)
        else:
            print("Sorry, I couldn't process that. Please try again.")
    return


if __name__ == "__main__":
    main()