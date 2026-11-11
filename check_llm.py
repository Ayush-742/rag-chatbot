import time
from rag import get_llm

llm = get_llm()
t = time.time()
reply = llm.invoke("Reply with the single word OK.")
print("seconds:", round(time.time() - t, 1), "| reply:", reply.content)