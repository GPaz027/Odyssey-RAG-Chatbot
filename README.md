# Challenge 04 (Basic RAG) - Gonzalo Paz

## Modules

* Components 1-3 (Text Q&A Chatbot): Main script in `interactive_odyssey_chatbot.py`
* Component 4 (Python Library Chatbot): Main script in `interactive_python_documentation_chatbot.py`

## Notebook

* The Jupyter Notebook (`notebooks/RAG-Practice.ipynb`) was used to develop step by step both solutions.
* You may try to run the Notebook, but it is better to run the apps directly from CLI.

### Text Q&A Chatbot

*  Text chosen: Odyssey (.txt file)

#### Sample Queries

* Who is Telemachus?
* Who is Penelope?
* Who is Athena helping in the story?
* How does Odysseus escape from the Cyclops?
* How does Odysseus reveal himself to the Cyclops?
* Why does Poseidon hate Odysseus?
* What happens when Odysseus reaches the island of the Sirens?
* How does Odysseus prove his identity at the end?
* What is Odysseus’ main struggle during his journey?
* What role do the gods play in Odysseus’ fate?
* **Who was Shakespeare?** -> Non-related question (the model does not answer)

### Python Library Chatbot

* Library chosen: Requests (quickstart & advanced sections)

#### Sample Queries

* What happens if multiple IP addresses exist for a domain name regarding timeouts?
* Does requests support HTTP/3?
* What is the difference between r.text and r.content?
* How do I disable redirects?
* How does SSL certificate verification work?
* How do I configure proxies?
* What are the differences between a normal request and a Session?
* **How do I use asyncio with requests?** -> Non-related question (the model does not answer)
* **Who created the requests library?** -> Non-related question (the model does not answer) 
* **What is the release date of requests 3.0?** -> Non-related question (the model does not answer)

## Getting started

* To install the dependencies, run `pip install -r requirements.txt`
* To run the Python Library Chatbot, run `python interactive_python_documentation_chatbot.py` on project root.
* To run the Odyssey chatbot, run `python interactive_odyssey_chatbot.py`.
* To exit the interface, just type `exit` on the command line.


