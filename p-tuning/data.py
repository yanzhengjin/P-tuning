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
from functools import partial

import paddle

from paddlenlp.dataaug import WordDelete, WordInsert, WordSubstitute, WordSwap
from paddlenlp.datasets import MapDataset, load_dataset


def extend_with_pseudo_data(data_ds, pseudo_path, labels_to_ids):
    """
    Extend train dataset with pseudo labeled examples if exists.
    """
    if pseudo_path is None:
        return data_ds
    with open(pseudo_path, "r", encoding="utf-8") as fp:
        pseudo_data = [json.loads(x.strip()) for x in fp]
    data_ds = MapDataset([x for x in data_ds] + pseudo_data)
    return data_ds
# 这段代码定义了一个函数extend_with_pseudo_data，用于将伪标签数据扩展到训练数据集中。
#
# 函数的输入参数包括：
#
# data_ds：原始的训练数据集，类型为MapDataset。
# pseudo_path：伪标签数据的文件路径，以JSON格式存储。
# labels_to_ids：标签到标签ID的映射字典。
# 首先，函数会检查是否提供了伪标签数据的路径。如果没有提供路径，则直接返回原始的训练数据集。
#
# 如果提供了伪标签数据的路径，函数会打开该文件，并将其解析为一个列表，列表中的每个元素是一个包含样本信息的字典，使用json.loads进行解析。
#
# 接下来，函数将原始训练数据集和伪标签数据合并到一个新的MapDataset中。这里使用了列表推导式来将原始数据集和伪标签数据合并到新列表中。
#
# 最后，函数返回扩展后的数据集。
#
# 总结来说，这个函数的作用是将伪标签数据扩展到原始的训练数据集中，将伪标签数据读取并合并到训练数据集中，生成一个新的数据集用于训练模型。

def extend_with_data_augment(data_ds, aug_type, num_aug=10, percent=0.1, aug_base="mlm", example_keys=None):
    """
    Extend train dataset with augmentation.
    """
    if example_keys is None:
        return data_ds
    if aug_type is None or aug_type == "None":
        return data_ds
    if aug_type == "delete":
        aug = WordDelete(create_n=num_aug, aug_percent=percent)
    elif aug_type == "substitute":
        aug = WordSubstitute(aug_base, create_n=num_aug, aug_percent=percent)
    elif aug_type == "insert":
        aug = WordInsert(aug_base, create_n=num_aug, aug_percent=percent)
    elif aug_type == "swap":
        aug = WordSwap(create_n=num_aug, aug_percent=percent)
    else:
        raise ValueError("Unsupported data augment strategy `{}`".format(aug_type))
    # 代码定义了一个函数extend_with_data_augment，用于将数据集进行数据增强。
    #
    # 函数的输入参数包括：
    #
    # data_ds：原始的数据集，类型为MapDataset。
    # aug_type：数据增强类型，指定要使用的数据增强策略。
    # num_aug：每个样本要生成的增强样本数量。
    # percent：用于数据增强的文本替换或删除的百分比。
    # aug_base：用于数据增强的基准文本，例如mlm表示使用掩码语言模型。
    # example_keys：要进行数据增强的样本键列表。
    # 首先，函数会检查是否提供了要进行数据增强的样本键列表，如果没有提供，则直接返回原始的数据集。
    #
    # 接下来，函数会根据提供的数据增强类型aug_type来选择相应的数据增强策略。根据不同的增强类型，将创建相应的数据增强器对象。
    #
    # 以下是不同增强类型对应的数据增强器对象：
    #
    # delete：使用WordDelete进行词语删除增强。
    # substitute：使用WordSubstitute进行词语替换增强。
    # insert：使用WordInsert进行词语插入增强。
    # swap：使用WordSwap进行词语交换增强。
    # 如果提供的增强类型不在上述四种类型中，会抛出ValueError异常。
    #
    # 最后，函数返回经过数据增强后的新数据集。
    #
    # 总结来说，这个函数的作用是根据指定的数据增强类型和参数对数据集进行增强，生成一定数量的增强样本，并将其合并到原始的数据集中，生成一个新的数据集用于训练模型。
    aug_data = []
    for example in data_ds:
        for key in example_keys:
            text_aug = aug.augment(example[key])
            for text in text_aug:
                new_example = example.copy()
                example[key] = text
                aug_data.append(new_example)

    data_ds = MapDataset([x for x in data_ds] + aug_data)
    return data_ds
    # 定义了一个空列表aug_data，用于存储增强后的数据。
    #
    # 接下来，使用嵌套的循环对原始数据集data_ds中的每个样本进行处理。外层循环遍历每个样本，内层循环遍历要进行增强的样本键example_keys。
    #
    # 在内层循环中，首先通过数据增强器对象aug对样本中指定的文本进行增强，生成增强后的文本列表text_aug。
    #
    # 然后，针对每个增强后的文本，创建一个新的样本new_example，通过复制原始样本example并替换指定键key的值为增强后的文本text。
    #
    # 将新的样本new_example添加到aug_data列表中。
    #
    # 完成所有样本的增强后，将增强后的数据aug_data与原始数据集data_ds进行合并，并将合并后的结果赋值给data_ds。
    #
    # 最后，将合并后的数据集data_ds作为函数的返回值返回给调用该函数的代码。
    #
    # 总结来说，这段代码的作用是根据提供的数据增强器对象、要增强的样本键和参数，对原始数据集进行增强，生成增强后的数据集，并返回增强后的数据集。

def convert_chid(data_ds):
    """
    Insert idioms into positions of `#idiom#` so that the task is converted
    to binary classification.
    """
    split_data_ds = []
    for example in data_ds:
        fragments = example["content"].split("#idiom#")
        label = example.get("answer", None)
        for index, cand in enumerate(example["candidates"]):
            new_example = {"content_pre": fragments[0], "content_post": fragments[1], "idiom": cand}
            if label is not None:
                new_example["label"] = str(int(index == label))
            split_data_ds.append(new_example)
    return MapDataset(split_data_ds)
# 代码实现了将任务转换为二分类任务的逻辑。
#
# 函数名convert_chid表明该函数用于转换CHID（中文幽默识别）任务。
#
# 函数接受一个数据集data_ds作为输入。
#
# 首先，定义了一个空列表split_data_ds，用于存储转换后的数据。
#
# 然后，通过遍历数据集中的每个样本example，将样本的内容根据#idiom#进行分割，得到前半部分和后半部分的片段。
#
# 同时，获取样本的标签（如果有）并记录为label变量。
#
# 接下来，对样本中的候选项candidates进行遍历，每次遍历得到一个候选项cand。
#
# 然后，创建一个新的样本new_example，其中包含前半部分片段content_pre、后半部分片段content_post和当前候选项idiom。
#
# 如果存在标签label，则将新样本的label设置为1（如果当前候选项索引与标签相等）或0（如果不相等）。
#
# 将新样本new_example添加到split_data_ds列表中。
#
# 完成所有样本的转换后，将转换后的数据集split_data_ds封装成MapDataset对象，并作为函数的返回值返回给调用该函数的代码。
#
# 总结来说，这段代码的作用是将CHID任务中的样本内容根据指定的分割符#idiom#进行分割，并将分割后的片段、候选项和标签（如果有）组合成新的样本，
# 最终生成一个转换后的数据集，并将其作为函数的返回值返回给调用该函数的代码。转换后的数据集用于进行二分类任务。

def convert_csl(data_ds):
    """
    Concatanate keywords and it can be replaced by keyword `options` in develop versioin.
    实现了将关键词连接起来的逻辑。
    """
    concat_data_ds = []
    for example in data_ds:
        example["keyword"] = "，".join(example["keyword"])
        concat_data_ds.append(example)
    return MapDataset(concat_data_ds)
# 代码实现了将关键词连接起来的逻辑。
#
# 函数名convert_csl表明该函数用于转换CSL（中文科学文献阅读理解）任务。
#
# 函数接受一个数据集data_ds作为输入。
#
# 首先，定义了一个空列表concat_data_ds，用于存储转换后的数据。
#
# 然后，通过遍历数据集中的每个样本example，将样本中的关键词列表example["keyword"]使用逗号进行连接，得到一个字符串形式的关键词。
#
# 将关键词连接后的结果存储到样本中的keyword键下。
#
# 将转换后的样本example添加到concat_data_ds列表中。
#
# 完成所有样本的转换后，将转换后的数据集concat_data_ds封装成MapDataset对象，并作为函数的返回值返回给调用该函数的代码。
#
# 总结来说，这段代码的作用是将CSL任务中的样本中的关键词列表进行连接，生成一个新的关键词字符串，并将其作为样本的一个键值对添加到转换后的数据集中，
# 最终生成一个转换后的数据集，并将其作为函数的返回值返回给调用该函数的代码。转换后的数据集用于进一步处理和分析。

def convert_cluewsc(data_ds):
    """
    Mark the pronoun and entity with special tokens.
    代码实现了对CLUEWSC（中文自然语言推理）任务中的数据进行转换，主要是标记代词和实体。

    函数名convert_cluewsc表明该函数用于转换CLUEWSC任务。

    函数接受一个数据集data_ds作为输入。

    首先，定义了一个空列表marked_data_ds，用于存储转换后的数据。

    然后，通过遍历数据集中的每个样本example，对样本进行处理。

    在处理过程中，会对样本中的代词和实体进行标记，具体标记方式可能根据具体任务和需求而有所不同。

    最终，将转换后的样本添加到marked_data_ds列表中。

    完成所有样本的转换后，将转换后的数据集marked_data_ds封装成MapDataset对象，并作为函数的返回值返回给调用该函数的代码。

    总结来说，这段代码的作用是对CLUEWSC任务中的样本进行转换，标记代词和实体，并生成一个转换后的数据集。转换后的数据集用于进一步处理和分析。具体的代词和实体标记方式需要根据任务需求来确定。
    """
    marked_data_ds = []
    for example in data_ds:
        target, text = example["target"], list(example["text"])
        pronoun, p_index = target["span2_text"], target["span2_index"]
        entity, e_index = target["span1_text"], target["span1_index"]
        label = example.get("label", None)
        # 遍历数据集data_ds中的每个样本，并提取相关信息。
        # 对于每个样本example，代码从中获取以下内容：
        # target: 表示目标对象，包含了目标的相关信息。
        # text: 表示文本内容，将其转换为列表形式。
        # 接下来，代码从target中提取了以下信息：
        # pronoun: 代词，表示需要标记的代词文本。
        # p_index: 代词在文本中的索引位置。
        # entity: 实体，表示需要标记的实体文本。
        # e_index: 实体在文本中的索引位置。
        # 最后，代码获取了样本的标签（label），如果标签不存在，则将其赋值为None。
        # 这段代码的作用是遍历数据集中的每个样本，并提取代词、实体、标签等相关信息。这些信息可能用于后续的处理和分析。
        if p_index > e_index:
            text.insert(p_index, "_")
            text.insert(p_index + len(pronoun) + 1, "_")
            text.insert(e_index, "[")
            text.insert(e_index + len(entity) + 1, "]")
        else:
            text.insert(e_index, "[")
            text.insert(e_index + len(entity) + 1, "]")
            text.insert(p_index, "_")
            text.insert(p_index + len(pronoun) + 1, "_")
            # 这段代码根据代词和实体在文本中的索引位置，将它们标记出来。具体地，根据索引位置的大小关系，按照一定的顺序在文本中插入特殊的标记符号。
            # 如果代词的索引位置（p_index）大于实体的索引位置（e_index），则按照以下顺序插入标记符号：
            # 在代词之前插入下划线符号 _。
            # 在代词之后的位置插入下划线符号 _。
            # 在实体之前插入方括号符号[。
            # 在实体之后的位置插入方括号符号]。
            # 如果代词的索引位置小于实体的索引位置，则按照以下顺序插入标记符号：
            # 在实体之前插入方括号符号[。
            # 在实体之后的位置插入方括号符号]。
            # 在代词之前插入下划线符号_。
            # 在代词之后的位置插入下划线符号 _。
            # 通过在文本中插入特殊标记符号，这段代码标记了代词和实体的位置，以便后续的处理和分析。
        new_example = {"text": "".join(text), "pronoun": pronoun, "entity": entity}
        if label is not None:
            new_example["label"] = label
        marked_data_ds.append(new_example)
    return MapDataset(marked_data_ds)
    # 根据标记了代词和实体位置的文本生成一个新的例子（new_example）。new_example是一个字典，包含以下键值对：
    # "text"：将标记后的文本字符列表通过"".join(text)合并为字符串。
    # "pronoun"：代词的文本内容。
    # "entity"：实体的文本内容。
    # 如果存在标签（label），则将标签作为键值对"label"的值添加到new_example中。
    # 最后，将所有生成的新例子（new_example）组成的列表存储在marked_data_ds中，并返回一个MapDataset对象，其中包含了经过标记的数据集。


def convert_labels_to_ids(example, orig_key, labels_to_ids, pop_keys=None):
    """
    Convert the keyword in datasets to `labels`.
    """
    if orig_key in example:
        example["label_ids"] = labels_to_ids[example.pop(orig_key)]
    if pop_keys is not None:
        for key in pop_keys:
            if key in example:
                example.pop(key)
    return example
    # 用于将数据集中的关键字（orig_key）转换为相应的标签ID（labels_to_ids）。
    #
    # 如果例子中存在orig_key键，将其对应的值从字典中取出，并使用labels_to_ids字典将其转换为标签ID。将转换后的标签ID存储在example字典中的"label_ids"键下，同时将原始的orig_key键从example字典中移除。
    #
    # 如果存在pop_keys参数，会对example字典中的每个键进行检查，如果键在pop_keys列表中，则将该键从example字典中移除。
    #
    # 最后，返回更新后的example字典。

def convert_ids_to_words(example, token_ids):
    """
    Convert label id to the first word in mapping from labels to words,
    the length of which should coincide with that of `mask` in prompt.
    """
    if "label_ids" in example:
        labels = paddle.index_select(token_ids, paddle.to_tensor(example.pop("label_ids")), axis=0).squeeze(0)
        example["labels"] = labels
    return example
# 用于将标签ID（"label_ids"）转换为对应的单词。
#
# 如果example字典中存在"label_ids"键，表示有标签ID需要转换。
# 首先，使用paddle.index_select函数从token_ids中选择与"label_ids"对应的索引，这样可以获取到对应的单词。
# 然后，将获取到的单词存储在"labels"键下，并将原始的"label_ids"键从example字典中移除。
#
# 最后，返回更新后的example字典。

def load_fewclue_dataset(args, verbalizer, example_keys=None):
    """
    Load fewclue datasets and convert them to the standard format of PET.
    """
    split_id = args.split_id
    splits = [f"train_{split_id}", f"dev_{split_id}", "test_public", "test"]
    # 代码是用于加载fewclue数据集并将其转换为PET（Pattern - Exploiting Training）的标准格式。
    #
    # args是一个参数对象，包含了加载数据集所需的配置和参数信息。
    # verbalizer是一个用于将标签转化为文本的对象或函数，用于将数据集中的标签转化为可读的文本形式。
    # example_keys是一个可选的参数，用于指定要包含在加载的数据集示例中的键列表。
    # splits是一个包含要加载的数据集划分的列表。具体来说：
    #
    # train_{split_id}表示训练集的特定划分，其中
    # {split_id}表示划分的标识符。
    # dev_{split_id}表示开发集的特定划分。
    # test_public表示公开测试集，通常用于模型评估。
    # test表示隐藏测试集，通常用于模型最终的评估和比赛。
    # 该函数的主要功能是根据指定的划分加载fewclue数据集，并将数据转换为PET模型的标准格式，以便进行模型训练和评估。
    if args.task_name == "cluewsc":
        train_ds, dev_ds, public_test_ds, test_ds = load_dataset("fewclue", name=args.task_name, splits=splits)
        unlabeled_ds = None
    else:
        splits.append("unlabeled")
        train_ds, dev_ds, public_test_ds, test_ds, unlabeled_ds = load_dataset(
            "fewclue", name=args.task_name, splits=splits
        )
    data_ds = [train_ds, dev_ds, public_test_ds, test_ds, unlabeled_ds]
    # 代码根据args.task_name的值加载fewclue数据集的不同划分，并将它们存储在对应的变量中。
    # 如果args.task_name为
    # "cluewsc"，则只加载训练集、开发集、公开测试集和隐藏测试集，并将未标记数据集（unlabeled_ds）设置为None。
    # 否则，除了加载训练集、开发集、公开测试集和隐藏测试集外，还加载未标记数据集，并将它存储在unlabeled_ds变量中。
    # 最后，将加载的数据集存储在data_ds列表中，列表的顺序依次为训练集、开发集、公开测试集、隐藏测试集和未标记数据集（如果存在）。这样，可以通过索引来访问不同划分的数据集。
    # Preprocess data for mask prediction task.
    if args.task_name == "chid":
        for index, sub_data_ds in enumerate(data_ds):
            data_ds[index] = convert_chid(sub_data_ds)
    elif args.task_name == "cluewsc":
        for index, sub_data_ds in enumerate(data_ds[:-1]):
            data_ds[index] = convert_cluewsc(sub_data_ds)
    elif args.task_name == "csl":
        for index, sub_data_ds in enumerate(data_ds):
            data_ds[index] = convert_csl(sub_data_ds)
    orig_key = "label"
    pop_keys = ["id"]
    # 代码根据args.task_name的值对数据集进行转换操作。
    # 如果args.task_name为"chid"，则对除未标记数据集（如果存在）之外的所有数据集应用convert_chid函数进行转换。
    # 如果args.task_name为"cluewsc"，则对除未标记数据集之外的所有数据集应用convert_cluewsc函数进行转换。
    # 如果args.task_name为"csl"，则对所有数据集应用convert_csl函数进行转换。
    # 接下来，将orig_key设置为"label"，将pop_keys设置为包含"id"的列表。
    # 这些参数将用于在每个示例中进行标签转换和键的删除操作。
    # 通过上述转换和处理，数据集data_ds中的每个子数据集都将被转换为新的格式或进行特定的标记处理，以适应相应的任务需求。
    if args.task_name == "tnews":
        orig_key = "label_desc"
        pop_keys = ["keywords", "label", "id"]
    elif args.task_name == "iflytek":
        orig_key = "label_des"
        pop_keys = ["id", "label"]
    elif args.task_name == "ocnli":
        pop_keys = ["level", "label0", "label1", "label2", "label3", "label4", "genre", "prem_id", "id"]
    convert_label = partial(
        convert_labels_to_ids, orig_key=orig_key, labels_to_ids=verbalizer.labels_to_ids, pop_keys=pop_keys
    )
    # 代码中，根据args.task_name的值设置orig_key和pop_keys变量的不同取值。
    # 如果args.task_name为"tnews"，则将orig_key设置为"label_desc"，将pop_keys设置为包含"keywords"、"label"和"id"的列表。
    # 如果args.task_name为"iflytek"，则将orig_key设置为"label_des"，将pop_keys设置为包含"id"和"label"的列表。
    # 如果args.task_name为"ocnli"，则将pop_keys设置为包含"level"、"label0"、"label1"、"label2"、"label3"、"label4"、"genre"、"prem_id" 和"id"的列表。
    # 然后，使用convert_labels_to_ids函数将标签转换为标识符，并传入相应的参数：orig_key为原始键名，labels_to_ids为语言表述器中的标签到标识符映射，pop_keys为要删除的键列表。
    # convert_label是一个部分函数，通过指定部分参数来创建一个新的函数，它将在后续的数据处理中用于将标签转换为标识符。
    for index, sub_data_ds in enumerate(data_ds):
        if sub_data_ds is not None:
            data_ds[index] = sub_data_ds.map(convert_label)
    # 使用enumerate函数遍历data_ds列表，并对每个子数据集（sub_data_ds）进行处理。
    # 首先，通过检查sub_data_ds是否为None来确保数据集存在。如果数据集存在，就调用sub_data_ds.map(convert_label)方法。
    # map方法用于对数据集中的每个示例应用给定的函数。在这里，convert_label函数被应用于子数据集中的每个示例，将标签转换为标识符。
    # 最后，将转换后的数据集赋值回原来的位置，以更新data_ds列表中的相应子数据集。
    # Extend train dataset with data augmentation and pseudo-label data.
    data_ds[0] = extend_with_data_augment(
        data_ds[0], args.augment_type, args.num_augment, args.word_augment_percent, args.augment_method, example_keys
    )
    data_ds[0] = extend_with_pseudo_data(data_ds[0], args.pseudo_data_path, verbalizer.labels_to_ids)

    dev_labels = [x["label_ids"] for x in data_ds[1]]
    test_labels = [x["label_ids"] for x in data_ds[2]]

    convert_fn = partial(convert_ids_to_words, token_ids=verbalizer.token_ids[:, 0, :])
    data_ds[:3] = [x.map(convert_fn) for x in data_ds[:3]]

    return data_ds, (dev_labels, test_labels)
    # 代码进行了一系列的数据处理和转换操作：
    #
    # data_ds[0] = extend_with_data_augment(...)：将数据集data_ds[0]通过数据增强方法扩展，使用给定的数据增强类型、数量、百分比和方法来生成新的示例。
    #
    # data_ds[0] = extend_with_pseudo_data(...)：将数据集data_ds[0]与伪标签数据进行扩展，使用给定的伪标签数据路径和标签到标识符的映射来生成新的示例。
    #
    # dev_labels和test_labels：提取数据集data_ds[1]和data_ds[2]中示例的标签标识符，分别存储在dev_labels和test_labels列表中。
    #
    # convert_fn = partial(convert_ids_to_words, token_ids=verbalizer.token_ids[:, 0, :])：创建了一个偏函数convert_fn，该函数将标签标识符转换为与标签到词汇的映射中的第一个词相对应。
    #
    # data_ds[:3] = [x.map(convert_fn) for x in data_ds[:3]]：将数据集data_ds的前三个子数据集分别应用convert_fn函数进行转换，将标签标识符转换为相应的词汇。
    #
    # 最后，返回经过处理和转换后的数据集data_ds，以及dev_labels和test_labels作为附加的元组数据。
