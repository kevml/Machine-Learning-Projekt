# Pakete laden, dazu gehören auch Python Pakete
library(stringr)
library(dplyr)
library(keras)
library(reticulate)
library(caret)
np <- import("numpy")
librosa <- import("librosa")
pd <- import("pandas")
random <- import("random")
keras <- import("keras")

### 1. Schritt: Namen der Sounddateien importieren
files <- fs::dir_ls(
  path = "C:/Users/Kevin/Desktop/Sounddateien_neu_wav/", 
  recurse = TRUE, glob = "*.wav")

### 2. Schritt: Dataframe erzeugen
Audiodaten <- tibble(
  fname = files, 
  class = fname %>% str_extract("Sounddateien_neu_wav/.*/") %>% 
    str_replace_all("Sounddateien_neu_wav/", "") %>%
    str_replace_all("/", ""),
  class_id = class %>% as.factor() %>% as.integer() - 1L)

### 3. Schritt: Dataframe mischen ("shuffle")
set.seed(42)
rows <- sample(nrow(Audiodaten))
Audiodaten_shuffled <- Audiodaten[rows,]

### 4. Schritt: Audiofile filtern und einlesen
library(tuneR, warn.conflicts = F, quietly = T) # "nice functions for reading and manipulating .wav files"
library(signal, warn.conflicts = F, quietly = T) # signal processing functions
library(oce, warn.conflicts = F, quietly = T) # image plotting functions and nice color maps
# Funktion zur Erstellung von Spektrogrammen
spectrogram <- function(data){
  sound = readWave(data)
  # extraxt signal
  snd = sound@left
  # determine duration # für mein CNN eigentlich nicht notwendig
  dur = length(snd)/sound@samp.rate
  # determine sample rate
  fs = sound@samp.rate
  # demean to remove DC offset
  snd = snd-mean(snd)
  nfft = 1024
  window = 256
  overlap = 128
  # create spectrogram
  spec = specgram(x = snd, n = nfft, Fs = fs, window = window, overlap = overlap)
  # discard phase information
  P = abs(spec$S) # abs() macht den Wert positiv und entfernt den komplexen Teil der Zahl
  # normalize
  wert = P/max(P)
  wert2 = wert[,1:100] # Da die Länge der Sounddateien unterschiedlich ist, müssen wir einen festen Wert der Zeitachse herausfiltern, um jede Datei einheitlich zu machen
  # convert to dB
  P = 10*log10(wert) # wir machen es aus dem Grund, weil wir die linearität rausschaffen wollen -> Konvertierung in dezibel
  t = spec$t
  # plot spectrogram
  imagep(x = t, y = spec$f, z = t(P), col = oce.colorsViridis,
        ylab = 'Frequency [Hz]', xlab = 'Time [s]', drawPalette = T, decimate = F)
return(wert2)}

### 5. Schritt: Daten aufbereiten
filenames <- list()
spec_array <- list()
for (i in Audiodaten_shuffled["fname"]){
  filenames = append(filenames,i)
  for (i in filenames){
    x = spectrogram(i)
    spec_array = append(spec_array,tuple(np$array(x)))}}

class_id <- list()
for (i in Audiodaten_shuffled["class_id"]){
  class_id = append(class_id,i)}
### 6. Schritt: Daten 
# Daten in Trainings- und Testdaten splitten
train = spec_array[1:240] # Verhältnis(80:20)
test = spec_array[241:300]
# Klassen
train_class = class_id[1:240]
test_class = class_id[241:300]

# Tuple
x_train <- tuple(train)
y_train <- tuple(train_class)
x_test <- tuple(test)
y_test <- tuple(test_class)

# Konvertierung in numpy array
x_train = np$array(x_train)
x_test = np$array(x_test)

# Reshape für CNN input
x_train = np$reshape(x_train, c(240L,512L,100L,1L))
x_test = np$reshape(x_test, c(60L,512L,100L, 1L))

# One-Hot encoding der Klassen
y_train = np$array(keras$utils$to_categorical(y_train, 6L))
y_test = np$array(keras$utils$to_categorical(y_test, 6L))

# Konvertierung in numpy array
y_train = np$array(y_train)
y_test = np$array(y_test)

### 7. Schritt: Convoluntional neural network
model <- keras_model_sequential()
model %>%  
  layer_conv_2d(input_shape = c(512, 100, 1), 
                filters = 32, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 256, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_flatten() %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 6, activation = 'softmax')

model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = "adam",
  metrics = 'accuracy')

model %>% fit(
  x = x_train,
  y = y_train,
  epochs = 100,
  batch_size = 1)

prediction <- as_tibble(model$predict(x_test)) # zu Dataframe konvertieren, für rename

prediction <- prediction %>%
  rename(Bier = V1,
         Fahrrad = V2,
         Gier = V3,
         Hammer = V4,
         Schnitzel = V5,
         Zitrone = V6)
prediction

Prediction <- apply(prediction, 1, which.max) - 1
Reference <- apply(y_test, 1, which.max) - 1

confusionMatrix(table(Prediction,Reference))

sound_neu = "C:/Users/Kevin/Desktop/Neuer Ordner/Schnitzel mit Pommes.wav"
sound_spec = spectrogram(sound_neu)
sound_spec = np$reshape(sound_spec, c(1L,512L,100L,1L))
prediction_neu <- model$predict(sound_spec)
prediction_neu <- apply(prediction_neu, 1, which.max) - 1
prediction_neu
