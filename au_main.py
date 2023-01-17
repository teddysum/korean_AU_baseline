import json
import os

import torch
import argparse
import torch.nn as nn
from tqdm import trange, tqdm
from transformers import XLMRobertaModel, AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW
from sklearn.metrics import f1_score


PADDING_TOKEN = 1
S_OPEN_TOKEN = 0
S_CLOSE_TOKEN = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: ', device)

special_tokens_dict = {
    'additional_special_tokens': ['&name&', '&affiliation&', '&social-security-num&', '&tel-num&', '&card-num&', '&bank-account&', '&num&', '&online-account&']
}


def jsonload(fname, encoding="utf-8"):
    with open(fname, encoding=encoding) as f:
        j = json.load(f)

    return j


# json 개체를 파일이름으로 깔끔하게 저장
def jsondump(j, fname):
    with open(fname, "w", encoding="UTF8") as f:
        json.dump(j, f, ensure_ascii=False)


# jsonl 파일 읽어서 list에 저장
def jsonlload(fname, encoding="utf-8"):
    json_list = []
    with open(fname, encoding=encoding) as f:
        for line in f.readlines():
            json_list.append(json.loads(line))
    return json_list

# jsonlist를 jsonl 형태로 저장
def jsonldump(j_list, fname):
    f = open(fname, "w", encoding='utf-8')
    for json_data in j_list:
        f.write(json.dumps(json_data, ensure_ascii=False)+'\n')


def parse_args():
    parser = argparse.ArgumentParser(description="unethical expression classifier using pretrained model")
    parser.add_argument(
        "--train_data", type=str, default="../data/nikluge-au-2022-train.jsonl",
        help="train file"
    )
    parser.add_argument(
        "--test_data", type=str, default="../data/nikluge-au-2022-test.jsonl",
        help="test file"
    )
    parser.add_argument(
        "--pred_data", type=str, default="./output/result.jsonl",
        help="pred file"
    )
    parser.add_argument(
        "--dev_data", type=str, default="../data/nikluge-au-2022-dev.jsonl",
        help="dev file"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-5
    )
    parser.add_argument(
        "--eps", type=float, default=1e-8
    )
    parser.add_argument(
        "--do_train", action="store_true"
    )
    parser.add_argument(
        "--do_eval", action="store_true"
    )
    parser.add_argument(
        "--do_test", action="store_true"
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=10
    )
    parser.add_argument(
        "--base_model", type=str, default="xlm-roberta-base"
    )
    parser.add_argument(
        "--model_path", type=str, default="./saved_models/default_path/"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output/"
    )
    parser.add_argument(
        "--do_demo", action="store_true"
    )
    parser.add_argument(
        "--max_len", type=int, default=256
    )
    parser.add_argument(
        "--classifier_hidden_size", type=int, default=768
    )
    parser.add_argument(
        "--classifier_dropout_prob", type=int, default=0.1, help="dropout in classifier"
    )
    args = parser.parse_args()
    return args


class SimpleClassifier(nn.Module):

    def __init__(self, args, num_label):
        super().__init__()
        self.dense = nn.Linear(args.classifier_hidden_size, args.classifier_hidden_size)
        self.dropout = nn.Dropout(args.classifier_dropout_prob)
        self.output = nn.Linear(args.classifier_hidden_size, num_label)

    def forward(self, features):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.output(x)
        return x


class UnethicalExpressionClassifier(nn.Module):
    def __init__(self, args, num_label, len_tokenizer):
        super(UnethicalExpressionClassifier, self).__init__()

        self.num_label = num_label
        self.xlm_roberta = XLMRobertaModel.from_pretrained(args.base_model)
        self.xlm_roberta.resize_token_embeddings(len_tokenizer)

        self.labels_classifier = SimpleClassifier(args, self.num_label)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.xlm_roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=None
        )

        sequence_output = outputs[0]
        logits = self.labels_classifier(sequence_output)

        loss = None

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_label),
                                                labels.view(-1))

        return loss, logits


def tokenize_and_align_labels(tokenizer, form, label, max_len):
    data_dict = {
        'input_ids': [],
        'attention_mask': [],
        'label': [],
    }
    tokenized_data = tokenizer(form, padding='max_length', max_length=max_len, truncation=True)
    data_dict['input_ids'].append(tokenized_data['input_ids'])
    data_dict['attention_mask'].append(tokenized_data['attention_mask'])
    data_dict['label'].append(label)

    return data_dict


def get_dataset(raw_data, tokenizer, max_len):
    input_ids_list = []
    attention_mask_list = []
    token_labels_list = []

    for utterance in raw_data:
        tokenized_data = tokenize_and_align_labels(tokenizer, utterance['input'], utterance['output'] , max_len)
        input_ids_list.extend(tokenized_data['input_ids'])
        attention_mask_list.extend(tokenized_data['attention_mask'])
        token_labels_list.extend(tokenized_data['label'])

    return TensorDataset(torch.tensor(input_ids_list), torch.tensor(attention_mask_list),
                         torch.tensor(token_labels_list))


def evaluation(y_true, y_pred):

    y_true = list(map(int, y_true))
    y_pred = list(map(int, y_pred))

    print('f1_score: ', f1_score(y_true, y_pred, average=None))
    print('f1_score_micro: ', f1_score(y_true, y_pred, average='micro'))


def train_unethical_expression_classifier(args=None):
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    print('train_unethical_expression_classifier')
    print('model would be saved at ', args.model_path)

    print('loading train data')
    train_data = jsonlload(args.train_data)
    dev_data = jsonlload(args.dev_data)

    print('tokenizing train data')
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print('We have added', num_added_toks, 'tokens')
    train_dataloader = DataLoader(get_dataset(train_data, tokenizer, args.max_len), shuffle=True,
                                  batch_size=args.batch_size)
    dev_dataloader = DataLoader(get_dataset(dev_data, tokenizer, args.max_len), shuffle=True,
                                batch_size=args.batch_size)

    print('loading model')
    model = UnethicalExpressionClassifier(args, 2, len(tokenizer))
    model.to(device)

    # print(model)

    FULL_FINETUNING = True
    if FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=args.eps
    )
    epochs = args.num_train_epochs
    max_grad_norm = 1.0
    total_steps = epochs * len(train_dataloader)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    epoch_step = 0

    for _ in trange(epochs, desc="Epoch"):
        model.train()
        epoch_step += 1
        total_loss = 0

        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            model.zero_grad()

            loss, _ = model(b_input_ids, b_input_mask, b_labels)

            loss.backward()

            total_loss += loss.item()

            # print('batch_loss: ', loss.item())

            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)
        print("Epoch: ", epoch_step)
        print("Average train loss: {}".format(avg_train_loss))

        if args.do_eval:
            model.eval()

            pred_list = []
            label_list = []

            for batch in dev_dataloader:
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch

                with torch.no_grad():
                    loss, logits = model(b_input_ids, b_input_mask, b_labels)

                predictions = torch.argmax(logits, dim=-1)
                pred_list.extend(predictions)
                label_list.extend(b_labels)

            evaluation(label_list, pred_list)

        if not os.path.exists(args.model_path):
            os.makedirs(args.model_path)

        model_saved_path = args.model_path + 'saved_model_epoch_' + str(epoch_step) + '.pt'
        torch.save(model.state_dict(), model_saved_path)

    print("training is done")


def test_unethical_expression_classifier(args):

    test_data = jsonlload(args.test_data)
    pred_data = jsonlload(args.pred_data)

    temp_ground_truth_dict = {}

    true_list = []
    pred_list = []

    # 데이터 list로 변경
    for data in test_data:
        if data['id'] in temp_ground_truth_dict:
            return {
                "error": "정답 데이터에 중복된 id를 가지는 경우 존재"
            }
        temp_ground_truth_dict[data['id']] = data['output']

    for data in pred_data:
        if data['id'] not in temp_ground_truth_dict:
            return {
                "error": "제출 파일과 정답 파일의 id가 일치하지 않음"
            }
        true_list.append(temp_ground_truth_dict[data['id']])
        pred_list.append(data['output'])

    evaluation(true_list, pred_list)


def separate_by_s_token(form):
    splited_temp_form = form.split('</s></s>')
    splited_temp_form[0] = splited_temp_form[0].split('<s>')[-1]
    splited_temp_form[-1] = splited_temp_form[-1].split('</s>')[0]

    for i in range(len(splited_temp_form)):
        splited_temp_form[i] = splited_temp_form[i].strip()

    return splited_temp_form


def demo_unethical_expression_classifier(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    test_data = jsonlload(args.test_data)

    model = UnethicalExpressionClassifier(args, 2, len(tokenizer))
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()


    for data in tqdm(test_data):
        tokenized_data = tokenizer(data['input'], padding='max_length', max_length=args.max_len, truncation=True)

        input_ids = torch.tensor([tokenized_data['input_ids']]).to(device)
        attention_mask = torch.tensor([tokenized_data['attention_mask']]).to(device)

        with torch.no_grad():
            _, logits = model(input_ids, attention_mask)
        predictions = torch.argmax(logits, dim=-1)
        data['output'] = int(predictions[0])

    jsonldump(test_data, args.output_dir + 'result.jsonl')


if __name__ == '__main__':

    args = parse_args()

    if args.do_train:
        train_unethical_expression_classifier(args)
    elif args.do_demo:
        demo_unethical_expression_classifier(args)
    elif args.do_test:
        test_unethical_expression_classifier(args)
