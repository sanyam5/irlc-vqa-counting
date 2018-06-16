# HowManQA
wget -P data  https://einstein.ai/research/interpretable-counting-for-visual-question-answering/HowMany-QA.zip
unzip data/HowMany-QA.zip -d data/how_many_qa

# Visual Genome
wget -P data https://visualgenome.org/static/data/dataset/question_answers.json.zip
unzip data/question_answers.json.zip -d data/how_many_qa

wget -P data https://visualgenome.org/static/data/dataset/image_data.json.zip
unzip data/image_data.json.zip -d data/how_many_qa

mv data/how_many_qa/question_answers.json data/how_many_qa/HowMany-QA/visual_genome_question_answers.json
mv data/how_many_qa/image_data.json data/how_many_qa/HowMany-QA/visual_genome_image_data.json
