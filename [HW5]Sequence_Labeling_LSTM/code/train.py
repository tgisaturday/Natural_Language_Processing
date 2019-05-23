import utils
import model

# data build for wordembeddings
utils.data_build()

# create instance of config
config = utils.Config()

# build model
model = model.NERmodel(config)
model.build()

# read datasets
dev   = utils.data_read(config.filename_dev, config.processing_word,
                  config.processing_tag, config.max_iter)
train = utils.data_read(config.filename_train, config.processing_word,
                  config.processing_tag, config.max_iter)

# train model
model.train(train, dev)
