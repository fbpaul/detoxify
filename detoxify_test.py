
from detoxify import Detoxify
# each model takes in either a string or a list of strings

# results = Detoxify('original').predict('example text')
# results = Detoxify('unbiased').predict(['example text 1','example text 2'])
# results = Detoxify('multilingual').predict(['example text','exemple de texte','texto de ejemplo','testo di esempio','texto de exemplo','örnek metin','пример текста'])
results = Detoxify('unbiased').predict(["我警告你，最好給我把錢還清，不然下次見到你，我一定會把你揍進醫院了，我說到做到，別懷疑。",
                                            "I warn you, you'd better pay me back, otherwise next time I see you, I will beat you into the hospital. I mean what I say, don't doubt it."])
print(results)