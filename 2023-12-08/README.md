# Riassunto su deep neural networks

## Problemi lineari e non lineari
Alcuni problemi possono essere risolti con modelli lineari, cioè semplicemente tirando una linea per predire un valore continuo (regressione lineare) o per separare due insiemi di dati (classificazione binaria).

![Problemi lineari](https://www.engineersgarage.com/wp-content/uploads/2022/03/TCH45-01.png)

Abbiamo visto che questi modelli sono molto semplici, e sono descritti da una funzione come:

ŷ = w<sub>1</sub>x<sub>1</sub> + w<sub>2</sub>x<sub>2</sub> + w<sub>0</sub>

Dove _ŷ_ è la previsione, le _x_ sono le varie feature e le _w_ i pesi (w<sub>0</sub> è il bias, a volte indicato come _b_).

Questa è la rappresentazione grafica di un modello lineare, che è anche chiamato [perceptron](https://en.wikipedia.org/wiki/Perceptrons_(book)).

![Modello lineare](https://www.knime.com/sites/default/files/styles/inline_small/public/1-intro-deep-neural-networks.png?itok=KAiCTWXU)

Ad esempio, la funzione logica OR, così come la conosciamo in informatica, può essere risolta da un modello lineare perché i suoi possibili risultati (_false_ e _true_) possono essere separati da una linea.

La funzione logica XOR, invece, non è un problema lineare.

![Problemi non lineari](https://res.cloudinary.com/practicaldev/image/fetch/s--NK5uRtLC--/c_imagga_scale,f_auto,fl_progressive,h_900,q_auto,w_1600/https://dev-to-uploads.s3.amazonaws.com/i/lkli02223oqhlac1jetz.png)

Se vogliamo costruire un modello che sia in grado di predire correttamente i risultati dello XOR, dobbiamo ricorrere alle deep neural network, cioè un modello formato da più livelli.

## Deep Neural Network

Una Deep Neural Network è appunto formata da più livelli.
- il livello degli **input**, cioè quello rappresentato dalle feature che diamo in pasto al modello;
- uno o più livelli cosiddetti **hidden**, che si trovano tra gli input e l'output;
- il livello di **output**, cioè quello che produce la previsione.

La seguente immagine mostra una deep neural network composta da un livello di input, uno hidden e uno di output. Si dice però che questa è una rete neurale da **2 livelli** perché il livello di input non si conteggia.

![Deep neural network](https://mvanderbroek.com/images/ann_model.png)

Un modello così però non è ancora in grado di predire i valori dello XOR. Infatti, aggiungere livelli in questo modo non risolve granché perché **la composizione di funzioni lineari è sempre una funzione lineare**. 

Abbiamo bisogno di introdurre una **funzione non lineare** in corrispondenza dei livelli hidden, che è anche chiamata **funzione di attivazione**.

Perciò, dopo aver calcolato i valori del livello hidden o del livello di output, li passiamo a questa funzione di attivazione che aggiungerà non-linearità al modello.

![Attivazione](https://assets-global.website-files.com/5d7b77b063a9066d83e1209c/60d24112aa6bf00ee6dc2642_Group%20805.jpg)

## Funzioni di attivazione

Ce ne sono varie tra cui scegliere:

![Funzioni di attivazione](https://www.researchgate.net/profile/Junxi-Feng/publication/335845675/figure/fig3/AS:804124836765699@1568729709680/Commonly-used-activation-functions-a-Sigmoid-b-Tanh-c-ReLU-and-d-LReLU.ppm)

- **Sigmoid** è perlopiù usata nel livello di output, per problemi di classificazione binaria (sì/no) perché restituisce un valore tra 0 e 1;
- **Softmax** è anch'essa usata nel livello di output per classificazione multiclasse;
- **Tanh** (tangente iperbolica), **ReLU** e **Leaky ReLU** sono tipicamente usate nei livelli hidden. Quella in assoluto più usata è **ReLU** perché è molto semplice da calcolare: `Math.Max(0, x)` e perciò velocizza l'apprendimento del modello.

> C'è una spiegazione approfondita sul come scegliere le funzioni di attivazione in [questo articolo](https://www.v7labs.com/blog/neural-networks-activation-functions).

Grazie alla funzione di attivazione, una deep neural network è ora in grado di risolvere lo XOR.

![ReLU](https://i.ytimg.com/vi/s7nRWh_3BtA/maxresdefault.jpg)

Il modello è ora in grado di separare i valori _true_ dai valori _false.

![separazione](https://www.oreilly.com/api/v2/epubs/9781492037354/files/assets/mlst_1006.png)

## Back propagation

Abbiamo visto che i modelli lineari apprendono per iterazioni successive.

Anche nelle deep neural networks è più o meno la stessa cosa: in ogni iterazione si verificano due fasi.

1. Si moltiplicano input e pesi, si sommano per ottere la previsione e si calcola il loss;
2. Usando il _gradient descent_, i pesi vengono aggiornati per ridurre il loss. 

![fasi](https://www.nomidl.com/wp-content/uploads/2022/04/image-8.png)

Questa seconda fase, in cui il gradient descent viene applicato per ogni livello, si chiama **backpropagation**. Infatti, l'aggiornamento dei pesi inizia dal livello di output e influenza tutti i livelli hidden precedenti.

![backpropagation](https://blog.paperspace.com/content/images/2022/07/backpropagation.png)

> Per decidere come modificare un peso, viene calcolata la derivata della funzione di attivazione. La derivata è appunto lo strumento matematico che serve a determinare se è il caso di aumentare o diminuire il peso.

# Embeddings

Una rete neurale con molti livelli è in grado di fare previsioni accurate per problemi non-lineari molto complessi. Nel caso dei Large Language Model, i livelli possono contenere migliaia di neuroni.

![llm](https://neuroflash.com/wp-content/uploads/2023/07/gpt-4-different-gpt-models.png)

In modelli così complessi, emergono funzionalità avanzate come la capacità di interpretare un testo e compiere dei ragionamenti. 

È normale quindi che si voglia dare in pasto al modello degli input multimediali: non solo testo ma anche immagini e audio.

> Abbiamo visto in precedenza che il testo [passa attraverso un tokenizer](../2023-11-24/README.md#tokenizer-delle-frasi) che usa un dizionario per produrre il vettore che viene dato in input alla rete neurale.

Il livello di input deve quindi contenere centinaia di migliaia di neuroni, affinché si possano passare al modello tutti i bit del contenuto originale.

La numerosità dei neuroni può essere un problemone perché il livello di input è full-connected con il primo livello hidden, e questo costerà molti calcoli durante l'apprendimento.

Quello che si cerca di fare è di creare degli **embeddings**, cioè una rappresentazione dell'input molto più concisa, che poi possa essere collegata al livello hidden. In questo modo si riduce drasticamente il numero di calcoli.

Nel seguente esempio, il livello di input (azzurro) contiene molti neuroni che poi vengono ricondotti a un embedding (verde) di soli 3 valori.
![embeddings](https://developers.google.com/static/machine-learning/crash-course/images/EmbeddingExample1-1.svg)

Come si vede, è l'embedding ad essere collegato al layer hidden (rosa).

Possiamo interpretare i valori dell'embedding come **coordinate spaziali**. Gli input concettualmente più simili tra loro si troveranno vicini nello spazio.

![embeddings](https://developers.google.com/static/machine-learning/crash-course/images/linear-relationships.svg)

Gli embeddings sono utili sia nel caso di input con vettori sparsi (testo) che con vettori densi (audio, immagini).