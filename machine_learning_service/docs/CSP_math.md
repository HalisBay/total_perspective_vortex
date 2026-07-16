# CSP (Common Spatial Patterns) — Adım Adım Matematiksel Gösterim

Bu doküman, `MyCSP` sınıfının matematiğini kovaryanstan başlayıp whitening, özayrışım (eigendecomposition), filtre çıkarımı ve öznitelik (feature) üretimine kadar **tek bir zincir** halinde anlatır.

Amaç: Kod ezberi değil. Her matrisin nereden geldiğini, neden o çarpımı yaptığımızı ve `np.linalg.eigh` çağrılarının aslında hangi soruyu cevapladığını net göstermek. Her bölümde önce **"ne yapıyoruz / neden"**, sonra **matematik**, sonra da mümkün olduğunca **elle çözülmüş küçük bir sayısal örnek** var.

> **Tek cümlelik özet:** CSP, iki sınıflı bir problemde öyle uzaysal filtreler bulur ki, filtrelenmiş sinyalin **varyansı (= gücü)** bir sınıfta maksimum, diğer sınıfta minimum olur. Bu iki uçtaki güç değerleri sınıflandırıcı için son derece ayırt edici hale gelir.

---

## İçindekiler

0. [Problem: CSP neyi çözüyor?](#0-problem-csp-neyi-çözüyor)
1. [Veri formatı ve notasyon](#1-veri-formatı-ve-notasyon)
2. [Kovaryans matrisleri (`calculate_covariance`)](#2-kovaryans-matrisleri-calculate_covariance)
3. [Ortak (composite) kovaryans](#3-ortak-composite-kovaryans)
4. [Asıl hedef: Rayleigh oranı ve genelleştirilmiş özdeğer problemi](#4-asıl-hedef-rayleigh-oranı-ve-genelleştirilmiş-özdeğer-problemi)
5. [Whitening (beyazlatma)](#5-whitening-beyazlatma)
6. [Beyazlatılmış uzayda sınıf kovaryansları](#6-beyazlatılmış-uzayda-sınıf-kovaryansları)
7. [S₁'in özayrışımı — özdeğerlerin anlamı](#7-s₁in-özayrışımı--özdeğerlerin-anlamı)
8. [Bileşen sıralama (`component_order`)](#8-bileşen-sıralama-component_order)
9. [Filtreler ve seçim](#9-filtreler-ve-seçim)
10. [`transform`: sinyalden özniteliğe](#10-transform-sinyalden-özniteliğe)
11. [Tek zincir — özet ve kod↔matematik haritası](#11-tek-zincir--özet-ve-kodmatematik-haritası)
12. [Ek: Neden whitening + özayrışım = genelleştirilmiş özdeğer?](#12-ek-neden-whitening--özayrışım--genelleştirilmiş-özdeğer)

---

## 0) Problem: CSP neyi çözüyor?

CSP'nin klasik doğduğu yer **BCI (Brain–Computer Interface)** / EEG'dir. Tipik senaryo *motor imagery*'dir: kişi "sol el" veya "sağ el" hareketini **hayal eder**, biz de kafa derisine yerleştirilmiş çok sayıda elektrottan (kanaldan) gelen sinyale bakarak hangisini hayal ettiğini kestirmeye çalışırız.

Fizyolojik gerçek şu: bir eli hayal etmek, karşı beyin yarım küresindeki duyusal-motor korteks üzerindeki **mu (~8–12 Hz)** ve **beta (~13–30 Hz)** bandı gücünü düşürür/artırır (ERD/ERS — *event-related de/synchronization*). Yani ayırt edici bilgi **belirli bir frekans bandındaki gücün, belirli kanallarda** değişmesidir.

İki temel gözlem:

- **Uzaysal filtre nedir?** Tek bir kanala bakmak yerine kanalların **ağırlıklı toplamına** bakarız: $z(t) = \mathbf{w}^\top \mathbf{x}(t)$. Buradaki $\mathbf{w}$ ağırlık vektörü bir "uzaysal filtre"dir — hangi kanalların ne kadar karışacağını söyler.
- **Neden varyans = güç?** Sinyal önceden band-pass filtrelendiği için (mu/beta) yaklaşık **sıfır ortalamalıdır**. Sıfır ortalamalı bir sinyalin **varyansı**, o bandttaki **ortalama gücüne** eşittir:

$$
\operatorname{Var}(z) = \mathbb{E}[z^2] - \underbrace{(\mathbb{E}[z])^2}_{\approx 0} \approx \mathbb{E}[z^2] = \text{band gücü}
$$

O zaman CSP'nin hedefi tek cümleyle şu olur:

> Öyle bir $\mathbf{w}$ bul ki, $z = \mathbf{w}^\top \mathbf{x}$ sinyalinin **varyansı Sınıf 1'de büyük, Sınıf 2'de küçük** olsun. Sonra da tam tersini yapan $\mathbf{w}$'leri bul. Bu iki uçtaki varyans değerleri, iki durumu birbirinden en iyi ayıran özniteliklerdir.

Aşağıdaki tüm matematik, bu "bir sınıfta varyansı büyüt, diğerinde küçült" cümlesini hayata geçirmenin şık bir yoludur.

---

## 1) Veri formatı ve notasyon

Kod, girdi olarak **3 boyutlu** bir dizi bekler:

```text
X.shape = (n_epochs, n_channels, n_times)   # (deneme sayısı, kanal, zaman)
y.shape = (n_epochs,)                        # sınıf etiketleri (tam olarak 2 sınıf)
```

- **epoch (deneme):** Tek bir denemeye ait sinyal parçası. Bir epoch, $C \times T$ boyutlu bir matristir ($C$ kanal, $T$ zaman örneği).
- **kanal (channel):** Bir elektrot. Uzaysal boyut budur; filtreler bu boyut üzerinde çalışır.
- **zaman (time):** Örnekleme noktaları.

Tek bir epoch'u şöyle gösterelim (her satır bir kanal, her sütun bir zaman anı):

$$
E =
\begin{bmatrix}
x_{1,1} & x_{1,2} & \dots & x_{1,T} \\
x_{2,1} & x_{2,2} & \dots & x_{2,T} \\
\vdots & \vdots & \ddots & \vdots \\
x_{C,1} & x_{C,2} & \dots & x_{C,T}
\end{bmatrix} \in \mathbb{R}^{C \times T}
$$

Kod başında iki doğrulama var; ikisi de CSP'nin varsayımlarını korur:

```python
if X.ndim != 3:                 # epoch × kanal × zaman yapısı şart
    raise ValueError(...)
b_classes = np.unique(y)
if len(b_classes) != 2:         # CSP tanım gereği iki-sınıflı (binary) bir yöntemdir
    raise ValueError(...)
```

CSP özünde **iki** kovaryans matrisini karşılaştırır; bu yüzden ikili yapı zorunludur. (Çok sınıf için one-vs-rest gibi uzantılar vardır ama bu implementasyon saf ikili haldir.)

---

## 2) Kovaryans matrisleri (`calculate_covariance`)

### Amaç

Bize lazım olan şey "hangi kanal, hangi kanalla birlikte, ne kadar birlikte değişiyor?" bilgisidir. Bunu veren nesne **kovaryans matrisi**dir. Her sınıf için ayrı bir kovaryans çıkaracağız: $C_1$ ve $C_2$. Bunlar sınıfların "uzaysal enerji dağılımının şeklini" tutar.

### Adım 2.1 — Tek epoch'un kovaryansı

Bir epoch $E \in \mathbb{R}^{C \times T}$ için (band-pass sonrası ~sıfır ortalamalı kabul edilir):

$$
\tilde{C} = E E^\top \in \mathbb{R}^{C \times C}
$$

Bu $C \times C$ matrisin $(i,j)$ elemanı, $i.$ ve $j.$ kanalın zaman boyunca çarpımlarının toplamıdır — yani kanallar arası ortak değişim. Köşegen ($i=i$) ise o kanalın kendi enerjisidir.

### Adım 2.2 — Trace ile normalizasyon (kritik ayrıntı)

```python
epoch_cov /= np.trace(epoch_cov)
```

Farklı denemeler farklı **toplam güçte** olabilir (kişi bir denemede daha güçlü sinyal üretmiştir, empedans değişmiştir, vs.). Biz mutlak gücü değil, **gücün kanallar arasındaki dağılım şeklini** istiyoruz. `trace` (izi), matrisin köşegen toplamıdır ve **toplam enerjiye** eşittir:

$$
\operatorname{trace}(E E^\top) = \sum_{i=1}^{C} (E E^\top)_{ii} = \text{epoch'un toplam enerjisi}
$$

Buna bölünce toplam enerji 1'e sabitlenir; geriye sadece **oran/şekil** kalır. Böylece güçlü bir deneme, ortalamayı haksızca domine edemez:

$$
C^{(\text{epoch})} = \frac{E E^\top}{\operatorname{trace}(E E^\top)}, \qquad \operatorname{trace}\!\left(C^{(\text{epoch})}\right) = 1
$$

### Adım 2.3 — Sınıf içi ortalama

Bir sınıfa ait tüm epoch kovaryanslarının ortalaması, o sınıfın kovaryans tahminidir:

$$
C_k = \frac{1}{N_k} \sum_{E \in \text{sınıf } k} \frac{E E^\top}{\operatorname{trace}(E E^\top)}, \qquad k \in \{1, 2\}
$$

Kod bire bir bunu yapar:

```python
for epoch in x_cl:
    epoch_cov = epoch @ epoch.T
    epoch_cov /= np.trace(epoch_cov)   # şekli koru, mutlak gücü at
    cov_mat += epoch_cov
cov_mat /= len(x_cl)                    # sınıf içi ortalama
```

### Elle çözülmüş örnek

2 kanal, $T=3$ zaman örneği, sınıf başına 2 epoch alalım. Sınıf 1'in iki kanalı **birlikte** hareket etsin (aynı yön), Sınıf 2'nin kanalları **ters** hareket etsin. Bu, kanonik CSP senaryosudur.

**Sınıf 1, birinci epoch:**

$$
E_{1a} =
\begin{bmatrix}
2.0 & -1.0 & 1.0 \\
1.8 & -0.9 & 0.9
\end{bmatrix}
$$

Önce $E_{1a} E_{1a}^\top$ (elle, satır × satır iç çarpımlar):

$$
(1,1) = 2\cdot2 + (-1)(-1) + 1\cdot1 = 6
$$

$$
(1,2) = 2\cdot1.8 + (-1)(-0.9) + 1\cdot0.9 = 3.6 + 0.9 + 0.9 = 5.4
$$

$$
(2,2) = 1.8\cdot1.8 + 0.9\cdot0.9 + 0.9\cdot0.9 = 3.24 + 0.81 + 0.81 = 4.86
$$

$$
E_{1a} E_{1a}^\top =
\begin{bmatrix}
6.00 & 5.40 \\
5.40 & 4.86
\end{bmatrix},
\qquad
\operatorname{trace} = 6 + 4.86 = 10.86
$$

Trace'e bölünce:

$$
C^{(1a)} =
\begin{bmatrix}
0.5525 & 0.4972 \\
0.4972 & 0.4475
\end{bmatrix}
$$

Aynı işlemi Sınıf 2'nin bir epoch'una ($E_{2a}=\begin{bmatrix}1 & -0.5 & 0.7\\ -1 & 0.5 & -0.7\end{bmatrix}$) uygularsak, kanallar tam ters işaretli olduğu için tertemiz bir sonuç çıkar:

$$
C^{(2a)} =
\begin{bmatrix}
0.5 & -0.5 \\
-0.5 & 0.5
\end{bmatrix}
$$

İki epoch'un ortalamasını alınca sınıf kovaryansları:

$$
C_1 =
\begin{bmatrix}
0.5496 & 0.4975 \\
0.4975 & 0.4504
\end{bmatrix},
\qquad
C_2 =
\begin{bmatrix}
0.5 & -0.5 \\
-0.5 & 0.5
\end{bmatrix}
$$

Dikkat: $C_1$'in **pozitif** korelasyonlu (kanallar birlikte), $C_2$'nin **negatif** korelasyonlu (kanallar ters) olduğu köşegen-dışı işaretlerinden okunuyor. CSP tam da bu farkı yakalayacak.

---

## 3) Ortak (composite) kovaryans

### Amaç

İki sınıfın **toplam** yayılımını temsil eden ortak bir referans uzayı istiyoruz. Bunu iki kovaryansı toplayarak elde ederiz:

$$
C_c = C_1 + C_2
$$

```python
cov_total = cov1 + cov2
```

Neden toplam? Çünkü birazdan bu $C_c$'yi kullanarak veriyi öyle bir uzaya taşıyacağız ki (whitening), o uzayda "toplam yayılım her yönde eşit" olacak. Eşit toplam yayılım zemininde, iki sınıf arasındaki **fark** çok daha net görünür hale gelir.

### Örnek

Yukarıdaki sayılarla:

$$
C_c = C_1 + C_2 =
\begin{bmatrix}
1.0496 & -0.0025 \\
-0.0025 & 0.9504
\end{bmatrix}
$$

(Köşegen-dışı ~0 çıkması bu özel sayıların bir tesadüfü; genelde öyle olmak zorunda değil.)

---

## 4) Asıl hedef: Rayleigh oranı ve genelleştirilmiş özdeğer problemi

Koddaki whitening + iki özayrışım hamlesinin **neden** yapıldığını anlamak için, önce çözmeye çalıştığımız problemi açıkça yazalım. "Sınıf 1'de varyansı büyüt, Sınıf 2'de küçült" isteği, bir **oran** (Rayleigh quotient) maksimizasyonudur:

$$
J(\mathbf{w}) = \frac{\mathbf{w}^\top C_1 \mathbf{w}}{\mathbf{w}^\top C_2 \mathbf{w}}
$$

- $\mathbf{w}^\top C_1 \mathbf{w}$ = $\mathbf{w}$ filtresi uygulandığında **Sınıf 1'deki varyans**.
- $\mathbf{w}^\top C_2 \mathbf{w}$ = aynı filtrenin **Sınıf 2'deki varyansı**.
- $J$'yi **maksimize** eden $\mathbf{w}$: Sınıf 1'i şişiren yön. $J$'yi **minimize** eden $\mathbf{w}$: Sınıf 2'yi şişiren yön.

Bu oranın ekstremumları, lineer cebirden bilinen **genelleştirilmiş özdeğer problemi**nin çözümüdür:

$$
C_1 \mathbf{w} = \lambda\, C_2 \mathbf{w}
$$

Eşdeğer ve daha kullanışlı biçimi, $C_c = C_1 + C_2$ ile yazılan halidir:

$$
C_1 \mathbf{w} = \lambda\, C_c \mathbf{w}, \qquad \lambda \in [0, 1]
$$

Buradaki $\lambda$'nın çok net bir yorumu var:

$$
\lambda = \frac{\mathbf{w}^\top C_1 \mathbf{w}}{\mathbf{w}^\top C_c \mathbf{w}} = \frac{\text{Sınıf 1 varyansı}}{\text{Toplam varyans}} = \textbf{o yöndeki gücün Sınıf 1'e düşen oranı}
$$

Yani $\lambda \approx 1$: yön neredeyse tamamen Sınıf 1'e ait. $\lambda \approx 0$: neredeyse tamamen Sınıf 2. $\lambda \approx 0.5$: iki sınıf da eşit → **ayırt edici değil**.

> **Kilit fikir:** Genelleştirilmiş özdeğer problemini doğrudan çözmek yerine, kod bunu **iki adıma** böler: önce *whitening* ile $C_c$'yi birim matrise çevirir, ardından **standart** (genelleştirilmiş değil) bir özayrışım yapar. 12. bölümde bunun neden birebir aynı problemi çözdüğünü ispatlıyoruz. Şimdilik bu iki adımı takip edelim.

---

## 5) Whitening (beyazlatma)

### Amaç

Veriyi öyle bir doğrusal dönüşümden geçirmek istiyoruz ki, dönüşüm sonrası **ortak kovaryans birim matris** olsun: $P C_c P^\top = I$. "Beyaz" ismi buradan gelir — beyaz gürültü gibi, her yönde eşit yayılım, yönler arası korelasyon yok. Bu zeminde iki sınıfın farkını okumak çok kolaylaşır.

### Adım 5.1 — $C_c$'nin özayrışımı

$C_c$ simetrik ve pozitif tanımlı olduğu için `eigh` ile ayrıştırılır:

$$
C_c = U \Lambda U^\top
$$

- $U$: ortonormal özvektörler (yayılımın **ana eksenleri**).
- $\Lambda = \operatorname{diag}(\lambda_1, \dots, \lambda_C)$: her eksendeki yayılım miktarı.

```python
eig_vals, eig_vecs = np.linalg.eigh(cov_total)
eig_vals = np.maximum(eig_vals, self.eps)   # sayısal koruma
```

`eps` neden? Bir sonraki adımda $1/\sqrt{\lambda}$ hesaplanacak. Kanal sayısı zaman örneğinden fazlaysa veya kanallar doğrusal bağımlıysa bazı $\lambda$'lar **0'a** çok yaklaşır → $1/\sqrt{0}$ patlar. `eps = 1e-9` ile taban koyarak bu bölmeyi güvene alıyoruz.

### Adım 5.2 — Whitening matrisi

$$
P = \Lambda^{-1/2} U^\top
$$

```python
whiten_cov_matx = np.diag(1.0 / np.sqrt(eig_vals)) @ eig_vecs.T
```

Bu $P$ iki iş yapar: $U^\top$ veriyi ana eksenlere **döndürür**, $\Lambda^{-1/2}$ ise her ekseni **kendi yayılımına bölerek** birim ölçeğe getirir. Sonuç:

$$
P C_c P^\top = \Lambda^{-1/2} U^\top (U \Lambda U^\top) U \Lambda^{-1/2}
= \Lambda^{-1/2} \Lambda \Lambda^{-1/2} = I
$$

(Ara adımda $U^\top U = I$ olduğunu kullandık.)

### Örnek + doğrulama

$C_c$'nin özayrışımından:

$$
\Lambda \approx \operatorname{diag}(0.9503,\; 1.0497), \qquad
U \approx
\begin{bmatrix}
-0.0253 & -0.9997 \\
-0.9997 & 0.0253
\end{bmatrix}
$$

Whitening matrisi:

$$
P \approx
\begin{bmatrix}
-0.0259 & -1.0255 \\
-0.9757 & 0.0247
\end{bmatrix}
$$

Doğrulama — gerçekten birim matris çıkıyor:

$$
P C_c P^\top \approx
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix} \checkmark
$$

---

## 6) Beyazlatılmış uzayda sınıf kovaryansları

### Amaç

Artık aynı whitening dönüşümünü **her iki sınıf kovaryansına** uygulayıp beyazlatılmış uzaydaki hallerine bakıyoruz:

$$
S_1 = P C_1 P^\top, \qquad S_2 = P C_2 P^\top
$$

```python
S1 = whiten_cov_matx @ cov1 @ whiten_cov_matx.T
```

(Kod yalnızca $S_1$'i açıkça hesaplar; $S_2$'ye ihtiyaç yok, çünkü birazdan göreceğimiz özellik onu gereksiz kılıyor.)

### Kilit özellik: $S_1 + S_2 = I$

Whitening'i $C_c = C_1 + C_2$ üzerine kurmuştuk, dolayısıyla:

$$
S_1 + S_2 = P(C_1 + C_2)P^\top = P C_c P^\top = I
$$

Bunun **muazzam** bir sonucu var. $S_1$ ve $S_2$ aynı özvektörleri paylaşır ve özdeğerleri **1'e tamamlanır**. Yani $S_1 = B D_1 B^\top$ ise, aynı $B$ ile $S_2 = B D_2 B^\top$ ve:

$$
D_1 + D_2 = I \;\Longrightarrow\; \lambda_i^{(1)} + \lambda_i^{(2)} = 1 \quad \forall i
$$

Somut anlamı: $S_1$'in özdeğeri 1'e yakın olan bir yön, **otomatik olarak** $S_2$'de 0'a yakındır. Yani "Sınıf 1'i en çok şişiren yön", aynı anda "Sınıf 2'yi en çok bastıran yöndür". CSP'nin tüm gücü bu tek eşitlikte saklı. Bu yüzden **tek bir özayrışım** ($S_1$'inki) iki ucu birden verir.

### Örnek + doğrulama

$$
S_1 \approx
\begin{bmatrix}
0.5005 & 0.5000 \\
0.5000 & 0.4996
\end{bmatrix},
\qquad
S_2 \approx
\begin{bmatrix}
0.4995 & -0.5000 \\
-0.5000 & 0.5004
\end{bmatrix}
$$

Toplamları gerçekten birim:

$$
S_1 + S_2 \approx
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix} \checkmark
$$

---

## 7) S₁'in özayrışımı — özdeğerlerin anlamı

### Amaç

Beyazlatılmış uzayda $S_1$'i ayrıştırıyoruz. Çıkan özvektörler bize (birazdan geri dönüştürünce) **CSP filtrelerini** verecek; özdeğerler ise her yönün ne kadar ayırt edici olduğunu.

$$
S_1 = B D B^\top
$$

```python
eig_vals, eig_vecs = np.linalg.eigh(S1)
```

### Özdeğerlerin yorumu

6. bölümdeki özellikten dolayı her $\lambda_i \in [0, 1]$ ve şu anlama gelir:

$$
\lambda_i = \text{o yöndeki gücün Sınıf 1'e düşen oranı}
$$

| $\lambda_i$ | Anlamı | Ayırt edicilik |
|---|---|---|
| $\approx 1$ | Yön neredeyse saf **Sınıf 1** varyansı | **Yüksek** |
| $\approx 0$ | Yön neredeyse saf **Sınıf 2** varyansı | **Yüksek** |
| $\approx 0.5$ | İki sınıf eşit güçte | **Sıfır** (işe yaramaz) |

Yani en değerli yönler özdeğer ekseninin **iki ucundaki** ($0$ ve $1$'e yakın) yönlerdir; ortadakiler ($0.5$) çöptür.

### Örnek

Bizim tasarladığımız "temiz ayrım" senaryosunda özdeğerler tam uçlara oturuyor:

$$
D = \operatorname{diag}(0.0,\; 1.0)
$$

$S_2$'nin karşılık gelen özdeğerleri ($1 - D$) beklendiği gibi $\operatorname{diag}(1.0,\; 0.0)$. Yani birinci yön **saf Sınıf 2**, ikinci yön **saf Sınıf 1**. Mükemmel ayrım.

---

## 8) Bileşen sıralama (`component_order`)

### Amaç

$S_1$'in özvektörlerini, **en ayırt edici olanlar en başa** gelecek şekilde sıralamak istiyoruz. Çünkü sonda `self.filters = W[:n_components]` diyerek sadece **baştaki** birkaçını tutacağız. Yanlış sıralama → değerli yönleri çöpe atmak demek. Kod, PCA gibi "sadece en büyük özdeğeri başa al" demez; MNE kütüphanesinin mantığını taklit eden iki strateji sunar.

### Strateji A — `"mutual_info"`

```python
idx = np.argsort(np.abs(eig_vals - 0.5))[::-1]
```

Fikir: bir yönün ayırt ediciliği, özdeğerinin **0.5'ten ne kadar uzak** olduğuyla ölçülür (7. bölümdeki tabloya bakın). O yüzden $|\lambda_i - 0.5|$ hesaplanır ve **azalan** sırayla dizilir — 0.5'ten en uzak (yani 0 veya 1'e en yakın) yönler en başa gelir.

İsmin "mutual information" olmasının sebebi: whitening sonrası bir bileşenin sınıf etiketiyle taşıdığı **karşılıklı bilgi**, tam olarak özdeğerin 0.5'ten uzaklığının artan bir fonksiyonudur. $\lambda = 0.5$ → bileşen sınıf hakkında **sıfır bilgi** taşır; $\lambda \to 0$ veya $\lambda \to 1$ → bilgi maksimum. Dolayısıyla $|\lambda - 0.5|$'e göre sıralamak, kabaca "en bilgilendirici bileşen önce" demektir.

**Örnek:** $\lambda = [0.0,\ 1.0]$ için $|\lambda - 0.5| = [0.5,\ 0.5]$ (ikisi de maksimum bilgi) → sıralama `idx = [1, 0]`, yani $[1.0,\ 0.0]$.

### Strateji B — `"alternate"`

```python
i = np.argsort(eig_vals)              # özdeğerleri küçükten büyüğe sırala
idx = np.empty_like(i)
idx[1::2] = i[: len(i) // 2]          # tek konumlara: en küçük yarı
idx[0::2] = i[len(i) // 2 :][::-1]    # çift konumlara: en büyük yarı (tersten)
```

Fikir: uçlardan içeriye doğru **büyük, küçük, büyük, küçük, …** diye diz. Böylece baştan $n$ tanesini aldığında, Sınıf-1 ağırlıklı ve Sınıf-2 ağırlıklı yönlerden **dengeli** bir karışım alırsın (klasik CSP'de filtreler hep uç çiftler halinde seçilir).

**Örnek** ($\lambda = [0.1, 0.3, 0.5, 0.7, 0.9]$, indeksler zaten sıralı $[0,1,2,3,4]$, $\text{yarı}=2$):

- $i[:2] = [0,1]$ → en küçük ikili (0.1, 0.3) → tek konumlara
- $i[2:] = [2,3,4]$, tersten $[4,3,2]$ → en büyük üçlü (0.9, 0.7, 0.5) → çift konumlara
- Sonuç `idx = [4, 0, 3, 1, 2]` → özdeğer dizisi $[0.9,\ 0.1,\ 0.7,\ 0.3,\ 0.5]$

Görüldüğü gibi: **en büyük, en küçük, 2. büyük, 2. küçük, orta** — tam bir zikzak. Ortadaki (0.5, işe yaramaz) en sona atılmış olur.

Son olarak seçilen sıralama özvektörlere uygulanır:

```python
eig_vecs = eig_vecs[:, idx]           # sütunları (özvektörleri) yeniden sırala
```

---

## 9) Filtreler ve seçim

### Amaç

$S_1$'in (sıralanmış) özvektörleri **beyazlatılmış** uzayda yaşıyor. Ama biz filtreleri **orijinal kanal** uzayında, ham sinyale doğrudan uygulanabilir halde istiyoruz. Bu yüzden whitening'i geri "içine katarız":

$$
W = B^\top P
$$

```python
W = eig_vecs.T @ whiten_cov_matx      # W = B^T P
self.filters = W[: self.n_components] # en ayırt edici n filtreyi tut
```

- $W$'nin **her satırı bir uzaysal filtre** $\mathbf{w}^\top$'dur.
- Neden $B^\top P$? 12. bölümdeki köprü şunu gösterir: beyazlatılmış uzaydaki özvektör $\mathbf{b}$ ile orijinal uzaydaki filtre arasında $\mathbf{w} = P^\top \mathbf{b}$ ilişkisi vardır, yani $\mathbf{w}^\top = \mathbf{b}^\top P$ — tam da $W$'nin bir satırı. Bu satırlar orijinal $J(\mathbf{w})$ Rayleigh oranının ekstremum noktalarıdır.
- $W[:n\_components]$ ile, 8. bölümde en başa taşınan **en ayırt edici** filtreleri tutar, gerisini atarız.

### Örnek

`mutual_info` sıralamasıyla:

$$
W \approx
\begin{bmatrix}
0.7080 & 0.7080 \\
0.6719 & -0.7423
\end{bmatrix}
$$

- **1. satır** (özdeğer 1'e karşılık, saf Sınıf 1 yönü): $\approx [0.708,\ 0.708]$ → iki kanalın **toplamı**. Mantıklı: Sınıf 1'de kanallar birlikte hareket ediyordu, toplamları büyük varyans verir.
- **2. satır** (özdeğer 0, saf Sınıf 2 yönü): $\approx [0.672,\ -0.742]$ → kanalların **farkı**. Sınıf 2'de kanallar ters hareket ediyordu; farkları o sınıfta büyük varyans verir.

Filtreler tam da veriyi ürettiğimiz yapıyı yakaladı.

---

## 10) `transform`: sinyalden özniteliğe

### Amaç

Öğrenilen filtreleri yeni bir epoch'a uygulayıp, sınıflandırıcının kullanacağı **sayısal öznitelik vektörünü** üretmek. Her filtre için tek bir sayı çıkar: o yöndeki güç (varyans).

### Adım 10.1 — Projeksiyon

$$
Z = W E \in \mathbb{R}^{n\_filters \times T}
$$

```python
csp_filter = self.filters @ sample
```

Her filtre, $C \times T$'lik epoch'u tek bir zaman serisine indirger (kanalları ağırlıklı toplar).

### Adım 10.2 — Güç (varyans)

$$
p_j = \frac{1}{T}\sum_{t=1}^{T} Z_{j,t}^2 = \overline{Z_j^2}
$$

```python
power = (csp_filter ** 2).mean(axis=1)
```

Zaman ekseni boyunca karelerin ortalaması. Sinyal ~sıfır ortalamalı olduğundan bu **varyansa** eşittir (0. bölüm). İşte CSP'nin baştan beri optimize ettiği "yöndeki güç" tam olarak bu.

### Adım 10.3 — (Opsiyonel) log

```python
if self.log:
    X_csp[i, :] = np.log(np.maximum(power, self.eps))
else:
    X_csp[i, :] = power
```

Güç değerleri pozitif ve çarpık (skewed) dağılır; **logaritma** bunları daha simetrik/Gaussvari hale getirir ki LDA gibi doğrusal sınıflandırıcılar daha rahat çalışsın. `np.maximum(power, eps)` ise $\log(0) = -\infty$ felaketini önler.

### Örnek — tek epoch

$E_{1a}$'yı öğrenilmiş filtrelerden geçirelim:

$$
Z = W E_{1a} \approx
\begin{bmatrix}
2.6904 & -1.3452 & 1.3452 \\
0.0078 & -0.0039 & 0.0039
\end{bmatrix}
$$

Zaman boyunca kare-ortalama:

$$
p \approx [\,3.6191,\; 0.0000\,]
\quad\Rightarrow\quad
\log p \approx [\,1.29,\; -10.41\,]
$$

Yani Sınıf-1 epoch'u, **1. filtrede** (Sınıf-1 yönü) yüksek güç, **2. filtrede** neredeyse sıfır güç üretti. Beklenen davranış bu.

### Bütün örneklerin öznitelikleri — ayrım gözle görülür

| Epoch | Gerçek sınıf | filtre 1 (log-güç) | filtre 2 (log-güç) |
|---|---|---|---|
| $E_{1a}$ | 1 | **+1.29** | −10.41 |
| $E_{1b}$ | 1 | **+1.49** | −7.95 |
| $E_{2a}$ | 2 | −20.72 | **+0.15** |
| $E_{2b}$ | 2 | −20.72 | **+1.46** |

Örüntü kristal netliğinde: **Sınıf 1** epoch'ları filtre 1'de yüksek / filtre 2'de dip; **Sınıf 2** epoch'ları tam tersi. Basit bir eşik bile bu iki sınıfı ayırır — CSP'nin işi budur.

---

## 11) Tek zincir — özet ve kod↔matematik haritası

Tüm akış, tek bir cümlenin uygulanışıdır: *"$C_c$'yi beyazlat, beyaz uzayda $S_1$'i ayrıştır, uçtaki yönleri filtre yap, güçlerini öznitelik olarak ver."*

```text
X (epoch×kanal×zaman)
   │  her epoch: E Eᵀ / trace(E Eᵀ)   → sınıf içi ortalama
   ▼
C₁ , C₂                                            (sınıf kovaryansları)
   │  topla
   ▼
C_c = C₁ + C₂                                      (ortak kovaryans)
   │  eigh → C_c = U Λ Uᵀ ;   P = Λ^(-1/2) Uᵀ
   ▼
WHITENING:  P C_c Pᵀ = I                           (ortak uzay birimlendi)
   │  S₁ = P C₁ Pᵀ    (ve gizliden S₂ = I − S₁)
   ▼
eigh(S₁) = B D Bᵀ ,   λ ∈ [0,1] = Sınıf-1 güç oranı
   │  0.5'ten uzak olanları başa sırala (mutual_info / alternate)
   ▼
W = Bᵀ P   →   filters = W[:n_components]          (uzaysal filtreler)
   │  yeni epoch: Z = W E ;  p = mean(Z², zaman) ;  (log)
   ▼
X_csp (epoch × n_components)                       (sınıflandırıcıya öznitelik)
```

Satır satır köprü:

| Kod | Matematik | Ne için |
|---|---|---|
| `epoch @ epoch.T` | $E E^\top$ | Kanallar arası ortak değişim |
| `/= np.trace(...)` | $\big/\operatorname{trace}(E E^\top)$ | Mutlak gücü at, şekli koru |
| `cov_mat /= len(x_cl)` | $\frac{1}{N_k}\sum$ | Sınıf kovaryansı $C_k$ |
| `cov_total = cov1 + cov2` | $C_c = C_1 + C_2$ | Ortak kovaryans |
| `eigh(cov_total)` | $C_c = U \Lambda U^\top$ | Yayılımın ana eksenleri |
| `np.maximum(vals, eps)` | $\lambda \leftarrow \max(\lambda, \varepsilon)$ | $1/\sqrt{\lambda}$ patlamasın |
| `diag(1/√vals) @ vecs.T` | $P = \Lambda^{-1/2} U^\top$ | Whitening matrisi |
| `P @ cov1 @ P.T` | $S_1 = P C_1 P^\top$ | Sınıf 1'i beyaz uzaya taşı |
| `eigh(S1)` | $S_1 = B D B^\top$ | Ayırt edici yönler + $\lambda$'lar |
| `argsort(|vals-0.5|)[::-1]` | $\downarrow |\lambda - 0.5|$ | En bilgilendirici yön başa |
| `eig_vecs.T @ whiten...` | $W = B^\top P$ | Filtreleri orijinal uzaya döndür |
| `W[:n_components]` | ilk $n$ satır | En iyi $n$ filtreyi tut |
| `filters @ sample` | $Z = W E$ | Projeksiyon |
| `(...**2).mean(axis=1)` | $\overline{Z_j^2}$ | Yön başına güç (varyans) |
| `np.log(max(power, eps))` | $\log p_j$ | Gaussvari öznitelik |

---

## 12) Ek: Neden whitening + özayrışım = genelleştirilmiş özdeğer?

4. bölümde asıl problemin şu Rayleigh oranı olduğunu söylemiştik:

$$
J(\mathbf{w}) = \frac{\mathbf{w}^\top C_1 \mathbf{w}}{\mathbf{w}^\top C_c \mathbf{w}}
$$

Şimdi **değişken değiştirelim**: $\mathbf{w} = P^\top \mathbf{v}$ olsun (yani beyaz uzaydaki bir $\mathbf{v}$ vektörünü orijinal uzaya taşıyoruz). Payı ve paydayı ayrı ayrı yazalım.

**Pay:**

$$
\mathbf{w}^\top C_1 \mathbf{w} = (P^\top \mathbf{v})^\top C_1 (P^\top \mathbf{v}) = \mathbf{v}^\top (P C_1 P^\top) \mathbf{v} = \mathbf{v}^\top S_1 \mathbf{v}
$$

**Payda:** (whitening'in kalbi burada devreye girer, $P C_c P^\top = I$)

$$
\mathbf{w}^\top C_c \mathbf{w} = \mathbf{v}^\top (P C_c P^\top) \mathbf{v} = \mathbf{v}^\top I \mathbf{v} = \mathbf{v}^\top \mathbf{v}
$$

Yerine koyunca:

$$
J = \frac{\mathbf{v}^\top S_1 \mathbf{v}}{\mathbf{v}^\top \mathbf{v}}
$$

Bu, karşımıza çıkabilecek **en temel** Rayleigh oranıdır ve ekstremumlarının $S_1$'in **özvektörleri**, ekstremum değerlerinin de $S_1$'in **özdeğerleri** olduğu lineer cebirin standart sonucudur. Yani:

- Beyaz uzaydaki en iyi yön $\mathbf{v} = \mathbf{b}$ (S₁'in bir özvektörü),
- Buna karşılık gelen orijinal filtre $\mathbf{w} = P^\top \mathbf{b}$, yani $\mathbf{w}^\top = \mathbf{b}^\top P$ — tam olarak `W = eig_vecs.T @ whiten_cov_matx` satırının bir satırı.

**Sonuç:** "Whitening yap, sonra $S_1$'in standart özayrışımını al" hamlesi, zor görünen genelleştirilmiş özdeğer problemini ($C_1 \mathbf{w} = \lambda C_c \mathbf{w}$) **birebir** çözer — sadece sayısal olarak çok daha stabil ve tek `eigh` çağrısıyla. Kodun tüm mimarisi bu tek özdeşliğin üstüne kurulu.

---

### Küçük notlar / sık takılınan yerler

- **Neden `eigh`, `eig` değil?** Kovaryanslar simetriktir; `eigh` bu yapıyı kullanır, gerçek özdeğerler döndürür ve daha hızlı/stabildir.
- **`eigh` özdeğerleri artan sırada verir.** Kod bu yüzden kendi sıralamasını (`argsort`) uygular; sırasız güvenmez.
- **`log=None` durumu.** Varsayılan `None` "yanlış" (falsy) sayılır, yani log **uygulanmaz**, ham güç döner. Log istiyorsan `log=True` vermelisin.
- **`n_components` çift seçmek** (klasik CSP'de 2–3 çift, yani 4–6) yaygındır: uçlardan hem Sınıf-1 hem Sınıf-2 baskın yönlerini dengeli almak için (özellikle `alternate` sıralamayla anlamlı).
