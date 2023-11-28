# Riassunto della call

Abbiamo fatto degli esercizi sulla classificazione multiclasse usando ML.NET (C#) e Axon (Elixir).

## Modelli ONNX

A prescindere dalle tecnologie usate per addestrare il modello, lo possiamo poi esportare nel formato open-source ONNX (Open Neural Network Exchange). Questo è un formato che rende il modello interoperable, cioè caricabile anche da librerie diverse da quelle che erano state usate per crearlo (es. creato con Tensorflow e poi usato da ML.NET).

> Abbiamo seguito l'[esempio completo](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-use-automl-onnx-model-dotnet) pubblicato nella documentazione Microsoft.

Caricare il modello ONNX è solo il primo passo: dobbiamo anche capire che tipo di input si aspetta di ricevere. Per questo possiamo usare il sito [Netron](https://netron.app/) (anche scaricabile come app) che ci mostrerà l'elenco di input e output, oltre che un diagramma della struttura del modello.

![https://cdn.linuxfordevices.com/wp-content/uploads/2023/08/Netron-in-action.png](https://cdn.linuxfordevices.com/wp-content/uploads/2023/08/Netron-in-action.png)


Il formato ONNX è adatto sia per i modelli di deep learning che per quelli di machine learning tradizionali, come ad esempio regressione lineare o logistica, e classificazione.

Su [huggingface.co](huggingface.co), un sito di collaborazione e condivisione di modelli, si trovano molti modelli nel formato ONNX. Mettono proprio a disposizione un filtro di ricerca. Vedi ad esempio quelli per la classificazione del testo.

https://huggingface.co/models?pipeline_tag=text-classification&library=onnx&sort=trending

## Classificazione multiclasse
Per il nostro esercizio, abbiamo scelto il modello [Finance Classification](https://huggingface.co/nickmuchi/sec-bert-finetuned-finance-classification) in grado di prendere in esame una frase a tema finanziario e determinarne il _sentiment_, cioè classificarla come:
 - **Bearish** (cioè con prospettiva tendente al ribasso);
 - **Neutra**;
 - **Bullish** (cioè con prospettiva tendente al rialzo).

 ![https://mentormecareers.com/wp-content/uploads/2023/02/Bullish-Vs-Bearish-1.png](https://mentormecareers.com/wp-content/uploads/2023/02/Bullish-Vs-Bearish-1.png)

 > Ad esempio, la frase `Burberry reports slow sales growth as Covid impact persists` viene classificata come `Bearish`.
 
 Trattandosi di 3 possibili classi, questo è quindi un modello di **classificazione multiclasse**.

## Softmax
I modelli di classificazione multiclasse restituiscono in output un vettore di _n_ numeri (in questo caso 3) di tipo `float`. Ogni numero rappresenta la probabilità di appartenenza alla relativa classe. Ad esempio:

```
[
    0.424389324,
    -1.593248923,
    2.8572409432
]
```

Come si vede, questi numeri sono valori grezzi, che non lasciano ben intuire quale sia l'effettiva probabilità di appartenenza a ciascuna delle 3 classi. Per avere un risultato più leggibile, i valori devono essere forniti alla **funzione Softmax** (è una cosiddetta *funzione di attivazione*), definita come segue:

``` csharp
float[] Softmax(float[] values)
{
    var maxVal = values.Max();
    var exp = values.Select(v => Math.Exp(v - maxVal));
    var sumExp = exp.Sum();

    return exp.Select(v => (float)(v / sumExp)).ToArray();
}
```

Il suo risultato sarà più leggibile:
```
[
    0.07984486,
    0.0106168995,
    0.90953827
]
```

Così si capisce molto meglio che c'è un 90% di probabilità che appartenga alla classe 3. La somma dei valori del vettore è esattamente 1.

![https://iq.opengenus.org/content/images/2019/01/softmax.png](https://iq.opengenus.org/content/images/2019/01/softmax.png)

**Softmax** è quindi la funzione di attivazione per i modelli di classificazione multiclasse, mentre invece **Sigmoid** lo era per i modelli di classificazione binaria. Sigmoid restituisce solo un numero, che rappresenta la probabilità di appartenenza alla classe positiva (es. se è spam).

![https://miro.medium.com/v2/resize:fit:1100/format:webp/1*WfERQXN_BtLi1eAh36_mvw.png](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*WfERQXN_BtLi1eAh36_mvw.png)

## Tokenizer delle frasi
Un modello di machine learning è in grado solo di macinare numeri. Quindi come è possibile dargli in input una frase, cioè una stringa di testo?

La stringa deve per prima cosa essere *tokenizzata*, cioè convertita in un vettore di numeri. Ogni numero rappresenta l'ID associato alla parola. Per questo il tokenizer usa un dizionario in cui ogni parola è associata a un proprio ID.

![https://miro.medium.com/v2/resize:fit:1400/1*f8NBZIutC7PuGo21rsVVkw.png](https://miro.medium.com/v2/resize:fit:1400/1*f8NBZIutC7PuGo21rsVVkw.png)

In realtà, non sempre c'è una corrispondenza 1:1 tra parola e numero. Il tokenizer può decidere di spezzare una parola in due o più frammenti oppure di aggiungere dei token speciali di separazione.

Nel caso illustrato qui in alto, la parola _rumination_ è stata spezzata in due parti: _rum_ e _ination_. Ecco un altro esempio in cui sono stati ottenuti due _wordpiece_ da una singola parola.

![https://vamvas.ch/assets/bert-for-ner/tokenizer.png](https://vamvas.ch/assets/bert-for-ner/tokenizer.png)

Il tokenizer di BERT, oltre al vettore di ID, produce anche due altri vettori chiamati _Attention mask_ e _Token type IDs_, il cui significato è [spiegato su Huggingface](https://huggingface.co/transformers/v3.2.0/glossary.html#attention-mask).

![https://heekangpark.github.io/assets/img/nlp/huggingface-bert-tokenizer.png](https://heekangpark.github.io/assets/img/nlp/huggingface-bert-tokenizer.png)

A noi non interessa molto quale sia il significato particolare di quei vettori perché in C# esiste il pacchetto [BertTokenizers](https://github.com/NMZivkovic/BertTokenizers) che fa il lavoro di tokenizzazione per noi. Può appunto essere usato per trasformare una stringa di testo nei vettori in questione che poi daremo in pasto tali e quali al modello ONNX.

## Deep learning

Abbiamo visto un esempio di deep learning con [Axon](https://hexdocs.pm/axon/Axon.html), l'equivalente di ML.NET per Elixir.

Permette di costruire una deep neural network in maniera molto leggibile, usando un'interfaccia fluente. A quel punto la si può addestrare dandogli in pasto i dataset di training e test.

Grazie al deep learning è possibile risolvere anche problemi non lineari, tipo risolvere il FizzBuzz. Anche in questo caso si tratta di classificazione multiclasse infatti, dato un numero, apparterrà a una di queste 4 classi:

 - Fizz
 - Buzz
 - FizzBuzz
 - Nessuna delle precedenti (perciò dovremo stampare il numero)

Anche nel caso del deep learning è estremamente importante saper selezionare le feature da dare in input al modello. Ad esempio, fornire il numero tale e quale non produrrà buoni risultati, soprattutto se la rete è molto piccola. Risultati decisamente più affidabili li otteniamo se gli diamo in pasto le seguenti feature:
 - Il resto della divisione intera per 3;
 - Il resto della divisione intera per 5;
 - Il resto della divisione intera per 15.

Nel corso della settimana approfondiremo il tema del deep learning.