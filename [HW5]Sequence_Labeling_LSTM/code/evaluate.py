# -*- coding: utf-8 -*-'
import model
import utils

# create instance of config
config = utils.Config()

# build model
model = model.NERmodel(config)
model.build()
model.restore_session(config.dir_model)

# create dataset
test  = utils.data_read(config.filename_test, config.processing_word,
                            config.processing_tag, config.max_iter)

# evaluate and interact
print ("Testing model over test set")
res = model.run_evaluate(test)
print("acc : "+str('%.2f'%res['acc'])+" - "+"f1 : "+str('%.2f'%res['f1']))

#input example
input_sent= ['23/SN', '일/NNB', '기성용/NNP', '의/JKG' ,'활약/NNG', '으로/JKB', '스완지시티/NNP', '는/JX', '리버풀/NNP', '전/NNG', '에서/JKB', '승리/NNG', '를/JKO', '얻/VV', '었/EP', '다/EC', './SF']

predicted_result = model.predict(input_sent)

for inseq, label in zip(input_sent, predicted_result) :
	print (inseq+" "+label)



