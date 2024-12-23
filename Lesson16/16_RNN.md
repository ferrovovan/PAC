# Рекуррентные нейронные сети

# Векторизация слов


```python
# N-граммы (комбинации из N последовательных терминов)
from sklearn.feature_extraction.text import CountVectorizer

print("Модель для N-грамм от 1 до 1 слова")  # bag of words
vect = CountVectorizer(ngram_range=(1, 1)) 
res = vect.fit_transform(['он не делает работу', 'не он делает работу']).toarray()
print(res)
print(vect.vocabulary_)

print()
print("Модель для N-грамм от 1 до 2 слов")
vect = CountVectorizer(ngram_range=(1, 2)) 
res = vect.fit_transform(['он не делает работу', 'не он делает работу']).toarray()
print(res)
print(vect.vocabulary_)
```

    Модель для N-грамм от 1 до 1 слова
    [[1 1 1 1]
     [1 1 1 1]]
    {'он': 2, 'не': 1, 'делает': 0, 'работу': 3}
    
    Модель для N-грамм от 1 до 2 слов
    [[1 1 1 1 0 1 0 1 1]
     [1 1 1 0 1 1 1 0 1]]
    {'он': 5, 'не': 2, 'делает': 0, 'работу': 8, 'он не': 7, 'не делает': 3, 'делает работу': 1, 'не он': 4, 'он делает': 6}


### Word2Vec
<img src="images/LessonsI/embeddings_structure.png" alt="w2v" height=50% width=50%>


```python
import codecs
import numpy as np
import gensim
```


```python
# with codecs.open('data/LOTR.txt', encoding='utf-8', mode='r') as f:
with codecs.open('/home/pakulich/Downloads/LOTR/LOTR.txt', encoding='utf-8', mode='r') as f:    
    docs = f.readlines()
    
max_sentence_len = 12

sentences = [sent for doc in docs for sent in doc.split('.')]
sentences = [[word for word in sent.lower().split()[:max_sentence_len]] for sent in sentences]
print(len(sentences), 'предложений')
```

    16168 предложений



```python
# Обучение модели
word_model = gensim.models.Word2Vec(sentences, vector_size=100, min_count=1, window=5, epochs=100)
```


```python
pretrained_weights = word_model.wv.vectors
vocab_size, embedding_size = pretrained_weights.shape
print(vocab_size, embedding_size)
```

    21571 100



```python
print('Похожие слова:')
for word in ['хоббит', 'кольцо', 'гном', 'эльф', 'лук', 'пин']:
    most_similar = ', '.join('%s (%.2f)' % (similar, dist) for similar, dist in word_model.wv.most_similar(word)[:8])
    print('  %s -> %s' % (word, most_similar))

```

    Похожие слова:
      хоббит -> приземистый (0.69), глупый (0.66), престарелый (0.64), румяным (0.64), длинноноги! (0.64), щеками, (0.61), завопил (0.60), колдун (0.60)
      кольцо -> оно (0.56), кольцо, (0.54), проклятье (0.50), таким, (0.47), сердце (0.46), найдено (0.46), келебримбор, (0.45), желание (0.44)
      гном -> встречей! (0.61), добрым (0.59), вперед! (0.57), голосом (0.57), голосом: (0.57), стойте (0.57), плечами (0.57), ура! (0.56)
      эльф -> свистящим (0.70), фермер, (0.68), эй! (0.66), собирая (0.65), леголас (0.64), голосом: (0.64), арагорн! (0.64), смеагорл, (0.64)
      лук -> колчан (0.65), одноэтажным, (0.65), светится (0.64), напитка (0.63), трон, (0.63), выточен (0.63), медовый (0.63), сырым, (0.63)
      пин -> мерри (0.70), бродяжник (0.62), сэм (0.60), пиппин (0.60), мерри, (0.59), леголас (0.55), пиппин, (0.55), халдир (0.55)



```python
# Получение вектора
word_model.wv.get_vector('хоббит')
```




    array([-0.5275733 , -0.49038413,  0.26530552, -0.55259633,  0.58883405,
           -0.5094188 ,  1.6025951 ,  1.8401952 , -1.0579935 ,  0.84028965,
            0.5595202 , -0.32262352, -0.32784835, -0.08210766, -0.33168343,
           -1.0980718 , -0.2449308 , -1.1738994 , -0.3005365 , -1.3311998 ,
            0.4990352 , -0.16485004, -0.23623984, -0.8880814 , -0.02566478,
           -0.9168636 ,  1.0509665 , -0.21874544, -0.35874528, -1.0377768 ,
           -0.04610081,  0.0591987 ,  0.41245377, -0.10338879,  0.18250333,
           -0.5635397 ,  0.41803986, -0.82597816,  0.17581068, -0.5880946 ,
           -0.16376743,  0.16672383, -0.70335317, -0.20451789,  0.5165798 ,
            0.5015588 ,  0.62059087,  0.7163039 ,  0.8874517 , -0.14966896,
           -0.4986672 , -0.9387815 ,  0.5809278 , -0.42801148, -0.41819337,
            0.66595894,  0.5018071 ,  0.54891986, -0.14095579,  0.8053368 ,
            1.332187  , -0.46000987,  2.1419404 , -0.86157966,  0.03455767,
            0.5401899 , -0.15926014,  0.24465467, -0.7497663 ,  0.07720115,
            0.0059006 ,  0.9634951 , -0.21999337,  0.8577407 ,  0.0361699 ,
           -0.3113782 ,  0.9843971 ,  0.68796545, -0.21583861, -0.3162096 ,
            0.8961235 ,  1.1203915 ,  0.05758803,  0.99355847, -0.16869242,
           -0.34653398,  0.8542577 , -0.57022184, -1.4141943 , -0.15982449,
            0.19826046,  0.19541225, -0.14984056,  0.72077584,  0.8022881 ,
            0.3373932 ,  0.14081486,  0.2796895 , -0.44255647, -0.30800146],
          dtype=float32)




```python
vec = word_model.wv.get_vector('пони') - word_model.wv.get_vector('хоббит') + word_model.wv.get_vector('маг')
word_model.wv.similar_by_vector(vec)
```




    [('пони', 0.8067206144332886),
     ('мерри', 0.5627685189247131),
     ('беспокойно', 0.49933090806007385),
     ('маг', 0.4963756501674652),
     ('пин', 0.48173031210899353),
     ('том', 0.4802573025226593),
     ('бродяжник', 0.4759186804294586),
     ('кухню', 0.4708172082901001),
     ('подъем', 0.4629760682582855),
     ('прямо', 0.46119844913482666)]



# RNN
Рекуррентные нейронные сети (RNN) — вид нейронных сетей, где связи между элементами образуют направленную последовательность. Благодаря этому появляется возможность обрабатывать серии событий во времени или последовательные пространственные цепочки. В отличие от многослойных перцептронов, рекуррентные сети могут использовать свою внутреннюю память для обработки последовательностей произвольной длины. Поэтому сети RNN применимы в таких задачах, где нечто целостное разбито на части, например: распознавание рукописного текста или распознавание речи. 


<img src="https://habrastorage.org/web/5c8/0fa/c22/5c80fac224d449209d888d18ea1111a8.png" alt="RNN" height=80% width=80%>

## Внутреннее устройство простых RNN
Входной вектор и вектор внутренней памяти объединяются и отправляются на полносвязный слой с функцией активации *tanh*


<img src="https://habrastorage.org/web/47d/ee6/2c3/47dee62c3af8498c946befa1f3330d90.png"  alt="RNN In"  height=65% width=65%>

$h_t = tanh(w * [h_{t-1}; x_t])$

где [h;x] - конкатенация векторов

## Виды RNN
<img src="images/rnn/rnns.jpg"  alt="RNN In"  height=70% width=70%>

### Many to many (Генерация текста)
<img src="images/rnn/rnn_m2m.jpeg"  alt="RNN In"  height=50% width=50%>

### One to many (Подпись изображений)
<img src="images/rnn/rnn_o2m.png"  alt="RNN In"  height=70% width=70%>
Началом и концом предложения являются специальные слова: $<start>$ $<end>$

# LSTM
Долгая краткосрочная память (Long short-term memory; LSTM) – особая разновидность архитектуры рекуррентных нейронных сетей, способная к обучению долговременным зависимостям.


<img src="https://habrastorage.org/web/67b/04f/73b/67b04f73b4c34ba38edfa207e09de07c.png" alt="LSTM"  height=80% width=80% >

* «вентиль забывания» контролирует меру сохранения значения в памяти
* «входной вентиль» контролирует меру вхождения нового значения в память
* «выходной вентиль» контролирует меру того, в какой степени значение, находящееся в памяти, используется при расчёте выходной функции активации для блока

Ключевые компоненты LSTM-модуля: состояние ячейки и различные фильтры. О состоянии ячейки можно говорить, как о памяти сети, которая передает соответствующую информацию по всей цепочке модулей. Таким образом, даже информация из ранних временных шагов может быть получена на более поздних, нивелируя эффект кратковременной памяти.

![image.png](data:image/png;base64,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)

По мере того, как происходит обучение, состояние ячейки изменяется, информация добавляется или удаляется из состояния ячейки структурами, называемыми фильтрами. Фильтры контролируют поток информации на входах и на выходах модуля на основании некоторых условий. Они состоят из слоя сигмоидальной нейронной сети и операции поточечного умножения.

### «Вентиль забывания»
#### Контролирует меру сохранения значения в памяти
<img src="https://habrastorage.org/web/a5f/31a/104/a5f31a104b184217aca105de9ab6d320.png">

### «Входной вентиль»
#### Контролирует меру вхождения нового значения в память
<img src="https://habrastorage.org/web/248/bf4/a75/248bf4a75ab74bf180b9c0e2e2cc5a58.png">

### «Выходной вентиль»
#### Контролирует меру того, в какой степени значение, находящееся в памяти, используется при расчёте выходной функции активации для блока
<img src="https://habrastorage.org/web/16d/5b5/783/16d5b5783ba34244afcf0f240133fb28.png">

Для замены старого состояния ячейки $𝐶_𝑡−1$ на новое состояние $𝐶_𝑡$. Необходимо умножить старое состояние на $𝑓_𝑡$, забывая то, что решили забыть ранее. Затем прибавляем 𝑖𝑡∗𝐶𝑡~. Это новые значения-кандидаты, умноженные на t – на сколько обновить каждое из значений состояния.

# GRU
Управляемые рекуррентные блоки (Gated Recurrent Units, GRU) — механизм вентилей для рекуррентных нейронных сетей, представленный в 2014 году. Было установлено, что его эффективность при решении задач моделирования музыкальных и речевых сигналов сопоставима с использованием долгой краткосрочной памяти (LSTM). По сравнению с LSTM у данного механизма меньше параметров, т.к. отсутствует выходной вентиль.


<img src="images/Lstm_gru.png" alt="GRU" height=60% width=60%>

# Архитектуры LSTM
### Объединение нескольких RNN
<img src="images/rnn/rnn_arch.png"  height=50% width=50%>

### Двунаправленные
<img src="images/rnn/rnn_bi.jpg"  height=70% width=70%>

### Seq2Seq
Sequence-to-sequence модель – это модель, принимающая на вход последовательность элементов (слов, букв, признаков изображения и т.д.) и возвращающая другую последовательность элементов.
<img src="images/rnn/seq2seq.png"  height=70% width=70%>

### Архитектура Google’s Neural Machine Translation
<img src="images/rnn/rnn_ggle.png"  height=100% width=100%>

### Механизм внимания (Attention)

— Техника используемая в рекуррентных нейронных сетях и сверточных нейронных сетях для поиска взаимосвязей между различными частями входных и выходных данных.

Слой механизма внимания представляет собой обычную, чаще всего однослойную, нейронную сеть на вход которой подаются $h_t$, t=1…m, а также вектор d в котором содержится некий контекст зависящий от конкретной задачи.

![image.png](data:image/png;base64,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)

$$c = ∑^m_{i=1}e_ih_i$$

Результатом работы слоя внимания является $c$ который, содержит в себе информацию обо всех скрытых состояниях $h_i$ пропорционально оценке $e_i$.

Таким образом при помощи механизма внимания достигается "фокусирование" декодера на определенных скрытых состояниях. В случаях машинного перевода эта возможность помогает декодеру предсказывать на какие скрытые состояния при исходных определенных словах на языке A необходимо обратить больше внимания при переводе данного слова на язык B. То есть на какие слова из исходного текста обратить внимание при переводе конкретного слова на язык назначения.

# Механизм внимания (Attention)
<img src="images/rnn/rnn_att.png"  height=30% width=30%>


<img src="images/rnn/att1.jpeg"  height=90% width=90%>
<img src="images/rnn/att2.jpeg"  height=90% width=90%>

### Результат
<img src="images/rnn/map_att.png"  height=70% width=70%>

# Пример


```python
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


df = pd.read_csv('./data/production.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>API</th>
      <th>Year</th>
      <th>Month</th>
      <th>Liquid</th>
      <th>Gas</th>
      <th>RatioGasOil</th>
      <th>Water</th>
      <th>PercentWater</th>
      <th>DaysOn</th>
      <th>_LastUpdate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5005072170100</td>
      <td>2014</td>
      <td>11</td>
      <td>9783</td>
      <td>11470</td>
      <td>1.172442</td>
      <td>10558</td>
      <td>1.079219</td>
      <td>14</td>
      <td>2016-04-06 17:20:05.757</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5005072170100</td>
      <td>2014</td>
      <td>12</td>
      <td>24206</td>
      <td>26476</td>
      <td>1.093778</td>
      <td>5719</td>
      <td>0.236264</td>
      <td>31</td>
      <td>2016-04-06 17:20:05.757</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5005072170100</td>
      <td>2015</td>
      <td>1</td>
      <td>20449</td>
      <td>26381</td>
      <td>1.290088</td>
      <td>2196</td>
      <td>0.107389</td>
      <td>31</td>
      <td>2016-04-06 17:20:05.757</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5005072170100</td>
      <td>2015</td>
      <td>2</td>
      <td>6820</td>
      <td>10390</td>
      <td>1.523460</td>
      <td>583</td>
      <td>0.085484</td>
      <td>28</td>
      <td>2016-04-06 17:20:05.757</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5005072170100</td>
      <td>2015</td>
      <td>3</td>
      <td>7349</td>
      <td>7005</td>
      <td>0.953191</td>
      <td>122</td>
      <td>0.016601</td>
      <td>13</td>
      <td>2016-06-16 14:07:33.203</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Подготовка данных по добыче
liquid = df.groupby('API')['Liquid'].apply(lambda df_: df_.reset_index(drop=True))
liquid.head()
```




    API             
    5005072170100  0     9783
                   1    24206
                   2    20449
                   3     6820
                   4     7349
    Name: Liquid, dtype: int64




```python
df_prod = liquid.unstack()
df_prod.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
      <th>23</th>
    </tr>
    <tr>
      <th>API</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5005072170100</th>
      <td>9783</td>
      <td>24206</td>
      <td>20449</td>
      <td>6820</td>
      <td>7349</td>
      <td>16552</td>
      <td>13844</td>
      <td>10655</td>
      <td>6135</td>
      <td>11105</td>
      <td>...</td>
      <td>6547</td>
      <td>5117</td>
      <td>5280</td>
      <td>4762</td>
      <td>4962</td>
      <td>4478</td>
      <td>4328</td>
      <td>4777</td>
      <td>3849</td>
      <td>3835</td>
    </tr>
    <tr>
      <th>5123377130000</th>
      <td>2341</td>
      <td>4689</td>
      <td>3056</td>
      <td>1979</td>
      <td>2037</td>
      <td>2260</td>
      <td>1961</td>
      <td>1549</td>
      <td>1364</td>
      <td>1380</td>
      <td>...</td>
      <td>898</td>
      <td>787</td>
      <td>880</td>
      <td>879</td>
      <td>773</td>
      <td>737</td>
      <td>543</td>
      <td>732</td>
      <td>559</td>
      <td>633</td>
    </tr>
    <tr>
      <th>5123379280000</th>
      <td>6326</td>
      <td>6405</td>
      <td>6839</td>
      <td>6584</td>
      <td>4775</td>
      <td>3917</td>
      <td>3840</td>
      <td>3031</td>
      <td>3137</td>
      <td>2669</td>
      <td>...</td>
      <td>1795</td>
      <td>1852</td>
      <td>1734</td>
      <td>1588</td>
      <td>1739</td>
      <td>1473</td>
      <td>1472</td>
      <td>1378</td>
      <td>1235</td>
      <td>1331</td>
    </tr>
    <tr>
      <th>5123379400000</th>
      <td>8644</td>
      <td>13977</td>
      <td>9325</td>
      <td>6445</td>
      <td>5326</td>
      <td>4538</td>
      <td>3403</td>
      <td>2534</td>
      <td>2685</td>
      <td>2597</td>
      <td>...</td>
      <td>1537</td>
      <td>1331</td>
      <td>1305</td>
      <td>1510</td>
      <td>1476</td>
      <td>1729</td>
      <td>1606</td>
      <td>1388</td>
      <td>1632</td>
      <td>814</td>
    </tr>
    <tr>
      <th>5123385820100</th>
      <td>1753</td>
      <td>4402</td>
      <td>1187</td>
      <td>1204</td>
      <td>1176</td>
      <td>1523</td>
      <td>1169</td>
      <td>782</td>
      <td>634</td>
      <td>597</td>
      <td>...</td>
      <td>60</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>236</td>
      <td>830</td>
      <td>6</td>
      <td>571</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>




```python
# Масштабирование и деление на трейн/тест
data = df_prod.values
data = data / data.max()
data = data[:, :, np.newaxis]

data_tr = data[:40]
data_tst = data[40:]
print(data_tr.shape, data_tst.shape)
```

    (40, 24, 1) (10, 24, 1)



```python
x_data = [data_tr[:, i:i+12] for i in range(11)]
y_data = [data_tr[:, i+1:i+13] for i in range(11)]

x_data = np.concatenate(x_data, axis=0)
y_data = np.concatenate(y_data, axis=0)
print(x_data.shape, y_data.shape)
```

    (440, 12, 1) (440, 12, 1)



```python
tensor_x = torch.Tensor(x_data) # transform to torch tensor
tensor_y = torch.Tensor(y_data)

oil_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
oil_dataloader = DataLoader(oil_dataset, batch_size=16) # create your dataloader
```


```python
for x_t, y_t in oil_dataloader:
    break
x_t.shape, y_t.shape
```




    (torch.Size([16, 12, 1]), torch.Size([16, 12, 1]))




```python
class OilModel(nn.Module):
    def __init__(self, timesteps=12, units=32):
        super().__init__()
        self.lstm1 = nn.LSTM(1, units, 2, batch_first=True)
        self.dense = nn.Linear(units, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        h, _ = self.lstm1(x)
        outs = []
        for i in range(h.shape[0]):
            outs.append(self.relu(self.dense(h[i])))
        out = torch.stack(outs, dim=0)
        return out
```


```python
model = OilModel()
opt = optim.Adam(model.parameters())
criterion = nn.MSELoss()
```


```python
NUM_EPOCHS = 20

for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times

    running_loss = 0.0
    num = 0
    for x_t, y_t in oil_dataloader:
        # zero the parameter gradients
        opt.zero_grad()

        # forward + backward + optimize
        outputs = model(x_t)
        loss = criterion(outputs, y_t)
        loss.backward()
        opt.step()

        # print statistics
        running_loss += loss.item()
        num += 1
        
    print(f'[Epoch: {epoch + 1:2d}] loss: {running_loss / num:.3f}')

print('Finished Training')
```

    [Epoch:  1] loss: 0.008
    [Epoch:  2] loss: 0.009
    [Epoch:  3] loss: 0.009
    [Epoch:  4] loss: 0.008
    [Epoch:  5] loss: 0.007
    [Epoch:  6] loss: 0.006
    [Epoch:  7] loss: 0.005
    [Epoch:  8] loss: 0.005
    [Epoch:  9] loss: 0.004
    [Epoch: 10] loss: 0.004
    [Epoch: 11] loss: 0.004
    [Epoch: 12] loss: 0.004
    [Epoch: 13] loss: 0.003
    [Epoch: 14] loss: 0.003
    [Epoch: 15] loss: 0.003
    [Epoch: 16] loss: 0.003
    [Epoch: 17] loss: 0.003
    [Epoch: 18] loss: 0.003
    [Epoch: 19] loss: 0.003
    [Epoch: 20] loss: 0.003
    Finished Training



```python
# Предскажем на год вперёд используя данные только первого года
x_tst = data_tst[:, :12]
predicts = np.zeros((x_tst.shape[0], 0, x_tst.shape[2]))

for i in range(12):
    x = np.concatenate((x_tst[:, i:], predicts), axis=1)
    x_t = torch.from_numpy(x).float()
    pred = model(x_t).detach().numpy()
    last_pred = pred[:, -1:]  # Нас интересует только последний месяц
    predicts = np.concatenate((predicts, last_pred), axis=1)
```


```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
for iapi in range(4):
    plt.subplot(2, 2, iapi+1)
    plt.plot(np.arange(x_tst.shape[1]), x_tst[iapi, :, 0], label='Actual')
    plt.plot(np.arange(predicts.shape[1])+x_tst.shape[1], predicts[iapi, :, 0], label='Prediction')
    plt.legend()
plt.show()
```


    
![png](images/16_RNN_47_0.png)
    


# Задание
1. Для обучения на нефтяных скважинах добавьте во входные данные информацию со столбцов Gas, Water (т.е. размер x_data будет (440, 12, 3)) и обучите новую модель. Выход содержит Liquid, Gas и Water (для дальнейшего предсказания). Графики с результатами только для Liquid.

# Лабораторная
1. Подготовьте данные для word2vec по одной из недавно прочитанных книг, удалив все символы, кроме букв и пробелов и обучите модель. Посмотрите результат.
2. Из этого же текста (п.1) возьмите небольшой фрагмент, разбейте на предложения с одинаковым числом символов. Каждый символ предложения закодируйте с помощью one hot encoding. В итоге у вас должен получиться массив размера (n_sentences, sentence_len, encoding_size).
3. На полученных в п.2 задании обучить модель RNN для предсказания следующего символа. Посмотрите результат при последовательной генерации.
4. На полученных в п.1 задании обучить модель RNN для предсказания следующего слова. Посмотрите результат при последовательной генерации.

