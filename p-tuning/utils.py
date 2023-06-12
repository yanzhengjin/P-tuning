# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import pathlib

import numpy as np
import paddle

from paddlenlp.datasets import load_dataset

# 这是一个标签映射字典（LABEL_TO_STANDARD），用于将特定任务的标签映射到标准标签（标准化的标签编码）。
# 字典的结构如下：
# "tnews"：一个任务名称，表示新闻分类任务。
# "news_story"至"news_game"：特定的新闻分类标签，例如"news_story"表示新闻故事类别。
# 每个特定标签都映射到一个标准标签编码，例如"news_story"映射到"100"。
# "iflytek"：一个任务名称，表示科大讯飞开放平台的任务。
# "打车"至"其他"：特定的任务标签，例如"打车"表示打车服务。
# 每个特定标签都映射到一个标准标签编码，例如"打车"映射到0。
# 这个字典的目的是为了将不同任务的具体标签映射到一个标准的编码，以方便在模型训练、评估和预测过程中进行统一处理。
LABEL_TO_STANDARD = {
    "tnews": {
        "news_story": "100",
        "news_culture": "101",
        "news_entertainment": "102",
        "news_sports": "103",
        "news_finance": "104",
        "news_house": "106",
        "news_car": "107",
        "news_edu": "108",
        "news_tech": "109",
        "news_military": "110",
        "news_travel": "112",
        "news_world": "113",
        "news_stock": "114",
        "news_agriculture": "115",
        "news_game": "116",
    },
    "iflytek": {
        "打车": 0,
        "美颜": 100,
        "影像剪辑": 101,
        "摄影修图": 102,
        "相机": 103,
        "绘画": 104,
        "二手": 105,
        "电商": 106,
        "团购": 107,
        "外卖": 108,
        "电影票务": 109,
        "社区服务": 10,
        "社区超市": 110,
        "购物咨询": 111,
        "笔记": 112,
        "办公": 113,
        "日程管理": 114,
        "女性": 115,
        "经营": 116,
        "收款": 117,
        "其他": 118,
        "薅羊毛": 11,
        "魔幻": 12,
        "仙侠": 13,
        "卡牌": 14,
        "飞行空战": 15,
        "射击游戏": 16,
        "休闲益智": 17,
        "动作类": 18,
        "体育竞技": 19,
        "地图导航": 1,
        "棋牌中心": 20,
        "经营养成": 21,
        "策略": 22,
        "MOBA": 23,
        "辅助工具": 24,
        "约会社交": 25,
        "即时通讯": 26,
        "工作社交": 27,
        "论坛圈子": 28,
        "婚恋社交": 29,
        "免费WIFI": 2,
        "情侣社交": 30,
        "社交工具": 31,
        "生活社交": 32,
        "微博博客": 33,
        "新闻": 34,
        "漫画": 35,
        "小说": 36,
        "技术": 37,
        "教辅": 38,
        "问答交流": 39,
        "租车": 3,
        "搞笑": 40,
        "杂志": 41,
        "百科": 42,
        "影视娱乐": 43,
        "求职": 44,
        "兼职": 45,
        "视频": 46,
        "短视频": 47,
        "音乐": 48,
        "直播": 49,
        "同城服务": 4,
        "电台": 50,
        "K歌": 51,
        "成人": 52,
        "中小学": 53,
        "职考": 54,
        "公务员": 55,
        "英语": 56,
        "视频教育": 57,
        "高等教育": 58,
        "成人教育": 59,
        "快递物流": 5,
        "艺术": 60,
        "语言(非英语)": 61,
        "旅游资讯": 62,
        "综合预定": 63,
        "民航": 64,
        "铁路": 65,
        "酒店": 66,
        "行程管理": 67,
        "民宿短租": 68,
        "出国": 69,
        "婚庆": 6,
        "工具": 70,
        "亲子儿童": 71,
        "母婴": 72,
        "驾校": 73,
        "违章": 74,
        "汽车咨询": 75,
        "汽车交易": 76,
        "日常养车": 77,
        "行车辅助": 78,
        "租房": 79,
        "家政": 7,
        "买房": 80,
        "装修家居": 81,
        "电子产品": 82,
        "问诊挂号": 83,
        "养生保健": 84,
        "医疗服务": 85,
        "减肥瘦身": 86,
        "美妆美业": 87,
        "菜谱": 88,
        "餐饮店": 89,
        "公共交通": 8,
        "体育咨讯": 90,
        "运动健身": 91,
        "支付": 92,
        "保险": 93,
        "股票": 94,
        "借贷": 95,
        "理财": 96,
        "彩票": 97,
        "记账": 98,
        "银行": 99,
        "政务": 9,
    },
}


def load_prompt_arguments(args):
    """
    Load prompt and label words according to prompt index.
    """
    with open(args.prompt_path, "r", encoding="utf-8") as fp:
        configs = json.load(fp)
        assert len(configs["verbalizer"]) == len(configs["template"])
        assert configs["verbalizer"][0] is not None
        verbalizer = [configs["verbalizer"][0]]
        last_verb_index = 0
        for index, verb in enumerate(configs["verbalizer"][1:]):
            if verb is None or len(verb) == 0:
                verbalizer.append(configs["verbalizer"][last_verb_index])
            else:
                verbalizer.append(verb)
                last_verb_index = index + 1
        configs["verbalizer"] = verbalizer
        args.prompt = configs["template"][args.prompt_index]["text"]
        label_words = configs["verbalizer"][args.prompt_index]
        if isinstance(label_words, list):
            label_words = {k: k for k in label_words}
        args.label_words = label_words
        return args
# 代码是一个函数 load_prompt_arguments，用于加载提示（prompt）和标签词（label words）。
#
# 函数的输入是一个参数对象 args，其中包含了一些配置信息，包括提示路径（args.prompt_path）和提示索引（args.prompt_index）等。
#
# 函数首先从指定的提示路径（args.prompt_path）加载配置文件，该配置文件是一个 JSON 文件，包含了提示和标签词的信息。然后，函数对加载的配置文件进行处理。
#
# 具体处理逻辑如下：
#
# 首先，函数检查加载的配置文件中的 "verbalizer" 和 "template" 字段的长度是否相同，确保它们具有相同的长度。
# 函数将 "verbalizer" 字段的第一个元素作为基准（即默认的标签词列表），并创建一个新的列表 verbalizer，将基准标签词添加到列表中。
# 函数遍历剩余的 "verbalizer" 元素，并根据是否为 None 或空列表来决定是否使用基准标签词。如果是 None 或空列表，则将基准标签词添加到列表中；否则，将当前的标签词列表添加到列表中，并更新基准标签词的索引。
# 函数将更新后的 verbalizer 列表重新赋值给配置文件中的 "verbalizer" 字段。
# 函数根据指定的提示索引（args.prompt_index）获取相应的提示文本，并将其赋值给 args.prompt。
# 函数根据指定的提示索引获取相应的标签词列表，并将其赋值给 args.label_words。如果标签词是一个列表，则将其转换为字典，其中键和值都是列表中的元素本身。
# 最后，函数返回更新后的参数对象 args。
# 这个函数的作用是根据指定的提示索引加载相应的提示文本和标签词，并将它们作为参数对象的属性进行返回。

def save_pseudo_data(save_path, task_name, label_preds, verbalizer, labels):
    """
    Combine unsupervised data and corresponding predicted labels and
    save one example per line.
    """
# 这段代码是一个函数
# save_pseudo_data，用于将伪标签数据和对应的预测标签组合起来，并将每个示例保存为一行文本。
#
# 函数的输入参数包括：
#
# save_path：保存文件的路径。
# task_name：任务名称。
# label_preds：标签预测结果。
# verbalizer：标签映射器。
# labels：标签列表。
# 函数的主要功能是将伪标签数据和对应的预测标签进行组合，并以一行文本的形式保存到指定的文件中。

    if task_name == "cluewsc":
        return None
    # "cluewsc" 是指 "CLUE Winograd模式挑战赛"（CLUE Winograd Schema Challenge）任务。它是由中文语言理解测评（CLUE）项目提出的一项自然语言处理任务，旨在测试模型对于理解具有歧义的语言结构的能力。
    # CLUE Winograd模式挑战赛的任务是解决一类句子中的指代消解问题，其中包含了一些具有歧义的语言结构，需要根据上下文来确定指代关系，从而选择正确的答案。这个任务对于模型的推理和语义理解能力提出了一定的挑战。
    # 参与CLUE Winograd模式挑战赛的模型需要对给定的句子进行理解和推理，判断句子中的指代关系，并给出正确的答案。

    data_ds = load_dataset("fewclue", name=task_name, splits="unlabeled")
    preds = paddle.to_tensor(label_preds.predictions)
    preds = verbalizer.aggregate_multiple_mask(preds)
    preds = paddle.nn.functional.softmax(preds, axis=1).numpy()
    label_preds = np.argmax(preds, axis=1)
    label_probs = np.max(preds, axis=1)
    pseudo_data = []
    for index, example in enumerate(data_ds):
        example["labels"] = labels[label_preds[index]]
        example["prob"] = str(label_probs[index])
        pseudo_data.append(example)
    save_data(pseudo_data, save_path)
    # 这段代码根据任务名称 task_name 的不同执行不同的操作。如果任务名称是 "cluewsc"，则返回 None，否则继续执行后续操作。
    #
    # 如果任务名称不是 "cluewsc"，则接下来的操作如下：
    #
    # 使用 load_dataset 函数从 "fewclue" 数据集中加载指定任务名称 task_name 的未标记数据集 "unlabeled"。
    # 将标签预测结果 label_preds 转换为PaddlePaddle张量。
    # 通过标签映射器 verbalizer 将多个掩码的标签预测结果聚合为一个张量。
    # 使用softmax函数对预测结果进行归一化，得到每个类别的概率。
    # 使用 numpy 函数将张量转换为NumPy数组，并使用 np.argmax 获取每个样本的最大概率对应的标签索引。
    # 使用 np.max 获取每个样本的最大概率值。
    # 创建一个空列表 pseudo_data。
    # 遍历未标记数据集中的每个样本，对于每个样本，将其预测标签和对应的概率值添加到样本中，并将样本添加到 pseudo_data 列表中。
    # 使用 save_data 函数将 pseudo_data 列表保存到指定的路径 save_path。
    # 总体上，这段代码的作用是将标签预测结果与未标记数据集中的样本进行匹配，并将匹配结果保存到文件中，以生成伪标签数据集。伪标签数据集包含了每个样本的预测标签及其概率值。

def save_fewclue_prediction(save_path, task_name, label_preds, verbalizer, labels):
    """
    Extract predicted labels and save as the format required by FewCLUE.
    """
    preds = paddle.to_tensor(label_preds.predictions)
    preds = verbalizer.aggregate_multiple_mask(preds)
# 这段代码的作用是将预测的标签转换为FewCLUE所要求的格式，并保存到文件中。
#
# 首先，将预测的标签转换为PaddlePaddle张量（paddle.to_tensor(label_preds.predictions)）。
# 然后，使用verbalizer.aggregate_multiple_mask方法对标签进行处理，将多个掩码对应的标签聚合为一个标签。这个方法可能用于处理多个掩码位置的情况，根据具体的实现细节来确定聚合方式。
# 接下来，将处理后的标签保存到指定的文件中。save_path参数指定了保存的路径，task_name参数表示任务名称，label_preds是包含预测结果的对象，
# verbalizer是用于标签转换的语言表述器，labels是用于将预测的标签映射回原始标签的字典或映射关系。
# 总的来说，这段代码的目的是将预测的标签按照FewCLUE规定的格式保存到文件中，以便后续的评估和分析。
    if task_name == "chid":
        batch_size = preds.shape[0]
        preds = paddle.nn.functional.softmax(preds, axis=1)[:, 1]
        preds = preds.reshape([batch_size // 7, 7])
    preds = paddle.nn.functional.softmax(preds, axis=1).numpy()
    preds = np.argmax(preds, axis=1)
    test_ds = load_dataset("fewclue", name=task_name, splits="test")
    # 这段代码根据任务名称对预测的标签进行处理，并加载测试数据集。
    #
    # 首先，通过判断任务名称是否为"chid"，来确定是否执行特定的处理逻辑。如果任务名称是"chid"，则进行以下操作：
    #
    # 获取预测结果的批大小（batch_size），然后使用paddle.nn.functional.softmax函数对预测结果进行softmax操作，按照第二个维度（轴）进行softmax，即对每个样本的预测结果进行归一化，得到概率分布。
    # 然后，选择第1列（索引为1）作为预测的结果，即选择预测为正例的概率。
    # 将处理后的预测结果进行形状重塑（reshape），将原本批大小的维度（batch_size）除以7，得到一个新的维度为7的形状，用于表示每个样本在7个类别上的预测概率。
    # 这可能是基于具体任务的数据集结构和标签编码方式而进行的特定处理。
    # 接下来，对预测结果进行再次的softmax操作，并将其转换为NumPy数组。然后，使用np.argmax函数在第1个维度上（轴）找到最大值的索引，即找到每个样本的最大预测概率对应的类别标签。
    # 最后，使用load_dataset函数加载指定任务和数据集类型（"test"）的测试数据集。这将返回一个表示测试数据集的对象，可以用于后续的评估或其他操作。
    # 综上所述，这段代码根据任务名称对预测的标签进行特定处理（针对"chid"任务），并加载与任务和数据集类型对应的测试数据集。
    ret_list = []
    maps = LABEL_TO_STANDARD.get(task_name, None)
    for idx, example in enumerate(test_ds):
        uid = example.get("id", idx)
        if task_name in ["bustm", "csl"]:
            ret_list.append({"id": uid, "label": str(preds[idx])})
        elif task_name == "chid":
            ret_list.append({"id": uid, "answer": preds[idx]})
        elif task_name in ["cluewsc", "eprstmt", "ocnli", "csldcp"]:
            ret_list.append({"id": uid, "label": labels[preds[idx]]})
        elif task_name in ["iflytek", "tnews"]:
            ret_list.append({"id": uid, "label": str(maps[labels[preds[idx]]])})
    save_file = task_name if task_name in ["bustm", "csldcp", "eprstmt"] else task_name + "f"
    save_data(ret_list, save_path, save_file + "_predict.json")
# 这段代码根据任务名称将预测结果转换为特定格式，并保存为JSON文件。
#
# 首先，定义一个空列表ret_list用于存储转换后的结果。
#
# 然后，通过LABEL_TO_STANDARD字典根据任务名称获取相应的标签映射（maps），如果不存在映射，则将maps设置为None。
#
# 接下来，使用enumerate函数遍历测试数据集中的每个样本，并获取样本的唯一标识符（uid）。根据任务名称执行以下操作：
#
# 如果任务名称是"bustm"或"csl"，将预测结果添加到ret_list中，以字典形式存储，包含键"id"和"label"，其中"label"的值为预测结果的字符串表示。
# 如果任务名称是"chid"，将预测结果添加到ret_list中，以字典形式存储，包含键"id"和"answer"，其中"answer"的值为预测结果。
# 如果任务名称是"cluewsc"、"eprstmt"、"ocnli"或"csldcp"，将预测结果添加到ret_list中，以字典形式存储，包含键"id"和"label"，其中"label"的值为预测结果对应的标签。
# 如果任务名称是"iflytek"或"tnews"，将预测结果添加到ret_list中，以字典形式存储，包含键"id"和"label"，其中"label"的值为预测结果对应的标签在映射（maps）中的值的字符串表示。
# 最后，根据任务名称确定要保存的文件名（save_file），如果任务名称是"bustm"、"csldcp"或"eprstmt"，则文件名为任务名称；
# 否则，在任务名称后面添加字母"f"作为文件名的一部分。将转换后的结果列表（ret_list）保存为JSON文件，文件名为"save_file + '_predict.json'"，并保存到指定的路径（save_path）中。
#
# 综上所述，这段代码根据任务名称将预测结果转换为特定格式，并将转换后的结果保存为JSON文件。保存的结果包含每个样本的唯一标识符和相应的预测标签或答案。

def save_data(data, save_path, save_file=None):
    if save_file is not None:
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
        save_path = os.path.join(save_path, save_file)
    with open(save_path, "w") as fp:
        for example in data:
            fp.write(json.dumps(example, ensure_ascii=False) + "\n")
# 这段代码用于将数据保存到文件中。
#
# 首先，如果提供了保存文件名（save_file），则使用pathlib.Path创建保存路径（save_path）的目录（包括所有父级目录），并确保目录存在。
# parents=True表示创建所有父级目录，exist_ok=True表示如果目录已经存在则不会引发错误。
#
# 接下来，使用os.path.join将保存路径（目录）和保存文件名连接起来，形成完整的保存文件路径。os.path.join函数会根据操作系统的不同，在路径中添加正确的分隔符。
#
# 接下来，使用open函数打开保存路径对应的文件，并以写入模式（"w"）打开文件。然后遍历数据列表（data）中的每个元素，对每个元素执行以下操作：
#
# 使用json.dumps函数将元素转换为JSON格式的字符串（确保不使用ASCII编码）。
# 将转换后的字符串写入文件中，并在末尾添加换行符。
# 综上所述，这段代码将数据列表中的每个元素转换为JSON格式，并将其逐行写入文件中，以保存数据。如果提供了保存文件名，则将数据保存到指定的文件中，否则将数据保存到指定的路径中。