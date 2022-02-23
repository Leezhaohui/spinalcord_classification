import os
import sys
sys.path.append(os.path.split(os.getcwd())[0])
sys.path.append(os.path.split(os.path.split(os.getcwd())[0])[0])
from keras import optimizers
from metrics import acc, sen, spe
from model import Net
from data_generator import train_generator, val_generator, train_batches, val_batches
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from data_generator import features, process_method, task
from keras.models import load_model

epochs = 500
log_path = "./tensorlog"
model_path = "./models"
model_ex_name = "_".join([task, features, process_method])

for path in [log_path, model_path]:
    if not os.path.exists(path):
        os.mkdir(path)

model = Net()
#model = load_model("models/spinal_os_5year_003.h5", compile=False)
model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=1e-4), metrics=[acc, sen, spe])

callback_lists = []
callback_lists.append(ModelCheckpoint(filepath="./models/%s_{epoch:03d}.h5" % model_ex_name,
                                      monitor="val_acc",
                                      verbose=1,
                                      save_best_only=True,
                                      mode="max"))
callback_lists.append(ReduceLROnPlateau(monitor="val_acc",
                                        factor=0.5,
                                        patience=20,
                                        min_lr=0,
                                        mode="max"))
callback_lists.append(TensorBoard(log_dir=log_path))

train_gen, val_gen = train_generator(), val_generator()
model.fit_generator(train_gen,
          steps_per_epoch=train_batches,
          epochs=epochs,
          validation_data=val_gen,
          validation_steps=val_batches,
          callbacks=callback_lists)
