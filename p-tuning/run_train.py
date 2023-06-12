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

import os
import time
from dataclasses import dataclass, field
from functools import partial

import paddle
from data import load_fewclue_dataset
from paddle.metric import Accuracy
from paddle.static import InputSpec
from utils import load_prompt_arguments, save_fewclue_prediction, save_pseudo_data

from paddlenlp.prompt import (
    MaskedLMVerbalizer,
    PromptModelForSequenceClassification,
    PromptTrainer,
    PromptTuningArguments,
    SoftTemplate,
)
from paddlenlp.trainer import PdArgumentParser
from paddlenlp.transformers import AutoModelForMaskedLM, AutoTokenizer
from paddlenlp.utils.log import logger


# yapf: disable
@dataclass
class DataArguments:
    task_name: str = field(default="eprstmt", metadata={"help": "The task name in FewCLUE."})
    split_id: str = field(default="0", metadata={"help": "The split id of datasets, including 0, 1, 2, 3, 4, few_all."})
    prompt_path: str = field(default="prompt/eprstmt.json", metadata={"help": "Path to the defined prompts."})
    prompt_index: int = field(default=0, metadata={"help": "The index of defined prompt for training."})
    augment_type: str = field(default=None, metadata={"help": "The strategy used for data augmentation, including `swap`, `delete`, `insert`, `subsitute`."})
    num_augment: str = field(default=5, metadata={"help": "Number of augmented data per example, which works when `augment_type` is set."})
    word_augment_percent: str = field(default=0.1, metadata={"help": "Percentage of augmented words in sequences, used for `swap`, `delete`, `insert`, `subsitute`."})
    augment_method: str = field(default="mlm", metadata={"help": "Strategy used for `insert` and `subsitute`."})
    pseudo_data_path: str = field(default=None, metadata={"help": "Path to data with pseudo labels."})
    do_label: bool = field(default=False, metadata={"help": "Whether to label unsupervised data in unlabeled datasets"})
    do_test: bool = field(default=False, metadata={"help": "Whether to evaluate model on public test datasets."})
    # 这是一个DataArguments类的定义，该类包含了用于配置数据处理的各种参数。
    # task_name：FewCLUE中的任务名称。
    # split_id：数据集的拆分ID，可以是0、1、2、3、4或few_all。
    # prompt_path：定义提示的路径。
    # prompt_index：用于训练的定义提示的索引。
    # augment_type：数据增强的策略，包括swap、delete、insert、subsitute等。
    # num_augment：每个示例的增强数据数量（仅在设置了augment_type时有效）。
    # word_augment_percent：序列中增强词汇的百分比（用于swap、delete、insert、subsitute）。
    # augment_method：insert和subsitute中使用的策略。
    # pseudo_data_path：带有伪标签的数据路径。
    # do_label：是否对无标签数据进行标记。
    # do_test：是否在公共测试数据集上评估模型。
    # 这些参数可以用于配置数据处理过程中的不同行为和选项。

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="ernie-1.0-large-zh-cw", metadata={"help": "Build-in pretrained model name or the path to local model."})
    export_type: str = field(default='paddle', metadata={"help": "The type to export. Support `paddle` and `onnx`."})
    dropout: float = field(default=0.1, metadata={"help": "The dropout used for pretrained model."})
# yapf: enable
# 这是一个ModelArguments类的定义，该类包含了用于配置模型的各种参数。
# model_name_or_path：预训练模型的名称或本地模型的路径。
# export_type：导出模型的类型。支持paddle和onnx。
# dropout：预训练模型中使用的dropout比例。
# 这些参数可以用于指定要使用的预训练模型以及相关的配置选项。例如，可以指定要使用的模型名称或本地模型的路径，并调整dropout的比例。

def main():
    # Parse the arguments.
    parser = PdArgumentParser((ModelArguments, DataArguments, PromptTuningArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    data_args = load_prompt_arguments(data_args)
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")
    paddle.set_device(training_args.device)
    # 这部分代码是用于解析命令行参数并将其存储在对应的数据类中。
    # 首先，创建一个PdArgumentParser对象，并指定要解析的参数类（ModelArguments、DataArguments和PromptTuningArguments）。
    # 然后，使用parser.parse_args_into_dataclasses()
    # 方法将命令行参数解析并存储在对应的数据类实例中，分别是model_args、data_args和training_args。
    # 接下来，调用load_prompt_arguments(data_args)
    # 函数，根据data_args中的参数加载和处理prompt相关的参数。
    # 然后，调用training_args.print_config()
    # 方法打印模型参数和数据参数的配置信息。
    # 最后，调用paddle.set_device()
    # 函数设置训练所使用的设备。
    # 这段代码的作用是解析命令行参数，并对模型参数、数据参数进行配置和打印相应的配置信息，并设置训练设备。
    # Load the pretrained language model.
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForMaskedLM.from_pretrained(
        model_args.model_name_or_path,
        hidden_dropout_prob=model_args.dropout,
        attention_probs_dropout_prob=model_args.dropout,
    )
    # 这部分代码用于加载预训练语言模型。
    # 首先，使用AutoTokenizer.from_pretrained()方法从预训练模型中加载tokenizer。model_args.model_name_or_path指定了预训练模型的名称或本地路径。
    # 接下来，使用AutoModelForMaskedLM.from_pretrained()
    # 方法从预训练模型中加载模型。model_args.model_name_or_path同样指定了预训练模型的名称或本地路径。额外的参数hidden_dropout_prob和attention_probs_dropout_prob用于设置模型中的dropout概率。
    # 这段代码的作用是加载预训练语言模型及其对应的tokenizer，并创建一个可用于生成预测的模型实例。

    # Define template for preprocess and verbalizer for postprocess.
    template = SoftTemplate(data_args.prompt, tokenizer, training_args.max_seq_length, model.get_input_embeddings())
    logger.info("Using template: {}".format(template.prompt))

    verbalizer = MaskedLMVerbalizer(data_args.label_words, tokenizer)
    labels_to_ids = verbalizer.labels_to_ids
    ids_to_labels = {idx: label for label, idx in labels_to_ids.items()}
    logger.info("Using verbalizer: {}".format(data_args.label_words))
    # 代码定义了预处理模板（template）和后处理的verbalizer。
    #在自然语言处理任务中，verbalizer是指将模型的输出转化为自然语言形式的过程。
    #在后处理阶段，模型通常会输出一些标识或者标签的编码，而verbalizer的作用就是将这些编码映射回原始的自然语言标签或者词汇。它定义了从模型输出到自然语言形式的转换规则或者映射关系。
    # 首先，使用SoftTemplate类创建了一个预处理模板。data_args.prompt指定了模板的路径，tokenizer是已加载的预训练模型的tokenizer，training_args.max_seq_length指定了输入序列的最大长度，model.get_input_embeddings()
    # 用于获取模型的输入嵌入。
    #
    # 接下来，使用MaskedLMVerbalizer类创建了一个后处理的verbalizer。data_args.label_words指定了标签词表的路径，tokenizer是已加载的预训练模型的tokenizer。
    #
    # 然后，verbalizer的labels_to_ids属性保存了标签到标签ID的映射，而ids_to_labels字典保存了标签ID到标签的映射。
    #
    # 最后，通过日志记录打印了使用的预处理模板和后处理的verbalizer的信息。
    #
    # 这段代码的作用是定义了预处理模板和后处理的verbalizer，用于数据的预处理和后处理过程。
    # Load datasets.
    data_ds, label_list = load_fewclue_dataset(data_args, verbalizer=verbalizer, example_keys=template.example_keys)
    train_ds, dev_ds, public_test_ds, test_ds, unlabeled_ds = data_ds
    dev_labels, test_labels = label_list
    # 在这段代码中，首先调用load_fewclue_dataset函数加载FewCLUE数据集并进行预处理，其中使用了之前定义的verbalizer和模板template。该函数返回了data_ds和label_list两个变量。
    #
    # data_ds是一个包含训练集、开发集、公共测试集、测试集和无标签数据集的列表。
    # 具体来说，data_ds中的第一个元素是训练集train_ds，第二个元素是开发集dev_ds，第三个元素是公共测试集public_test_ds，第四个元素是测试集test_ds，第五个元素是无标签数据集unlabeled_ds。
    # 这些数据集经过预处理，转换为了标准的PET格式。
    #
    # label_list是一个包含开发集和测试集的标签列表。
    # 具体来说，label_list中的第一个元素是开发集的标签列表dev_labels，第二个元素是测试集的标签列表test_labels。
    # 这些标签列表用于后续的评估和分析任务。
    #
    # 通过返回这些数据集和标签列表，可以在训练和评估过程中使用它们来加载数据、计算指标等操作。

    # Define the criterion.
    criterion = paddle.nn.CrossEntropyLoss()
    # 在这段代码中，criterion被定义为paddle.nn.CrossEntropyLoss()，它是交叉熵损失函数的实例化对象。
    #
    # 交叉熵损失函数是一种常用的损失函数，特别适用于多分类任务。它在训练分类模型时被广泛使用。对于每个样本，交叉熵损失函数计算模型预测值与真实标签之间的差异，并返回一个标量值作为损失值。
    #
    # 在使用交叉熵损失函数进行训练时，通常需要将模型的输出与真实标签传递给损失函数的forward方法，它会自动计算并返回损失值。然后可以使用该损失值进行反向传播和参数更新，以优化模型的性能。
    #
    # 在这个场景中，criterion用于计算模型的输出与真实标签之间的交叉熵损失，并在训练过程中使用该损失进行模型优化。

    # Initialize the prompt model with the above variables.
    prompt_model = PromptModelForSequenceClassification(
        model, template, verbalizer, freeze_plm=training_args.freeze_plm, freeze_dropout=training_args.freeze_dropout
    )
    # PromptModelForSequenceClassification是一个用于序列分类任务的模型。在这里，它被称为
    # "Prompt Model"，它是基于预训练语言模型（Pretrained Language Model, PLM）的模型。
    # 参数model是一个预训练语言模型，如BERT、GPT等。该模型已经通过无监督的方式在大规模文本数据上进行了预训练，具有很强的语言理解能力。
    #
    # 参数template是一个模板，用于将输入序列与特定的问题或任务相关信息结合起来。模板中包含了一些特殊标记或占位符，可以将问题或任务信息嵌入到输入序列中，以指导模型进行分类任务。
    #
    # 参数verbalizer是一个用于将模型的输出转化为可读性更好的文本表示的工具。它将模型的预测结果映射到特定的标签或类别，并提供了一些自然语言的描述。
    #
    # freeze_plm和freeze_dropout是用于控制是否冻结部分模型参数的标志。通过冻结部分参数，可以保持预训练语言模型的权重不变，只训练模型的特定部分，以适应特定的分类任务。
    #
    # 综上所述，PromptModelForSequenceClassification是一个结合了预训练语言模型、模板和文本转化工具的模型，用于序列分类任务。
    # 它通过将任务相关信息嵌入到输入序列中，并利用预训练语言模型的语言理解能力，实现对输入序列的分类预测。

    # Define the metric function.
    def compute_metrics(eval_preds, labels, verbalizer):
        metric = Accuracy()
        predictions = paddle.to_tensor(eval_preds.predictions)
        predictions = verbalizer.aggregate_multiple_mask(predictions)
        correct = metric.compute(predictions, paddle.to_tensor(labels))
        metric.update(correct)
        acc = metric.accumulate()
        return {"accuracy": acc}
    # compute_metrics是一个计算评估指标的函数。在这个函数中，使用了Accuracy作为评估指标，用于计算预测结果的准确率。
    #
    # 参数eval_preds是评估过程中的预测结果，包括predictions和label_ids。predictions是模型的预测输出，是一个张量。labels是标签数据，是一个列表或数组。
    #
    # 参数verbalizer是用于将模型的预测结果转化为可读性更好的文本表示的工具。
    #
    # 在函数中，首先将eval_preds.predictions转化为张量形式，并使用verbalizer对多个掩码进行聚合，得到最终的预测结果。然后，使用Accuracy计算预测结果的准确率。最后，返回包含准确率的字典，以便进行后续的评估分析。
    #
    # 综上所述，compute_metrics函数用于计算评估指标，其中使用了准确率作为评估指标，并利用verbalizer对预测结果进行转化和聚合。函数返回包含准确率的字典作为评估结果。

    # Initialize the trainer.
    dev_compute_metrics = partial(compute_metrics, labels=dev_labels, verbalizer=verbalizer)
    trainer = PromptTrainer(
        model=prompt_model,
        tokenizer=tokenizer,
        args=training_args,
        criterion=criterion,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        callbacks=None,
        compute_metrics=dev_compute_metrics,
    )
    # dev_compute_metrics是一个使用compute_metrics函数的部分应用（partial application）。
    # 在这里，dev_compute_metrics固定了labels参数为dev_labels，verbalizer参数为verbalizer，并将其作为参数传递给compute_metrics函数。
    #
    # trainer是一个PromptTrainer对象，用于训练和评估模型。它接受以下参数：
    #
    # model：PromptModelForSequenceClassification，要训练和评估的模型。
    # tokenizer：模型使用的分词器。
    # args：训练参数和设置。
    # criterion：损失函数，这里使用的是CrossEntropyLoss。
    # train_dataset：训练数据集，即train_ds。
    # eval_dataset：评估数据集，即dev_ds。
    # callbacks：回调函数，用于在训练过程中执行额外的操作，这里设为None。
    # compute_metrics：用于计算评估指标的函数，这里使用dev_compute_metrics。
    # 通过创建PromptTrainer对象，可以对模型进行训练和评估，并在训练过程中计算和输出评估指标。

    # Traininig.
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    time_stamp = time.strftime("%m%d-%H-%M-%S", time.localtime())
    # 如果training_args.do_train为True，则执行训练过程。
    #
    # 首先，调用trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    # 开始训练。resume_from_checkpoint参数指定是否从之前的检查点继续训练。
    #
    # 接下来，将训练结果保存，包括训练过程中的指标（metrics）。调用trainer.save_model()
    # 保存模型，trainer.log_metrics("train", metrics)
    # 记录训练指标的日志，trainer.save_metrics("train", metrics)
    # 保存训练指标，trainer.save_state()
    # 保存训练状态。
    #
    # 最后，使用time.strftime函数获取当前时间的时间戳，用于后续命名和记录训练过程中的输出。

    # Test.
    if data_args.do_test and public_test_ds is not None:
        test_compute_metrics = partial(compute_metrics, labels=test_labels, verbalizer=verbalizer)
        trainer.compute_metrics = test_compute_metrics
        test_ret = trainer.predict(public_test_ds)
        trainer.log_metrics("test", test_ret.metrics)
        # 这段代码片段用于在进行模型测试（evaluation）时计算指标（metrics）并记录结果。
        #
        # 首先，判断是否需要进行测试（do_test）以及是否存在公共测试数据集（public_test_ds）。如果条件满足，就会执行以下操作：
        #
        # 定义测试时使用的指标计算函数：使用partial函数创建了一个新的函数test_compute_metrics，该函数是在compute_metrics函数的基础上固定了部分参数，包括标签（test_labels）和verbalizer。
        # 这个函数将在后续的测试中用于计算指标。
        #
        # 将trainer对象的compute_metrics属性设置为测试指标计算函数：将之前定义的test_compute_metrics函数赋值给trainer对象的compute_metrics属性。
        # 这样在后续的测试过程中，trainer会使用该函数来计算测试指标。
        #
        # 进行测试预测：通过调用trainer对象的predict方法对公共测试数据集（public_test_ds）进行预测。预测结果保存在test_ret变量中。
        #
        # 记录测试指标：通过调用trainer对象的log_metrics方法，将测试指标（test_ret.metrics）记录下来。这些指标可以包括准确率（accuracy）等评估模型性能的指标。
        #
        # 这段代码片段的作用是在进行模型测试时计算指标并记录测试结果，以评估模型在公共测试数据集上的性能。

    # Predict.
    if training_args.do_predict and test_ds is not None:
        pred_ret = trainer.predict(test_ds)
        logger.info("Prediction done.")
        predict_path = os.path.join(training_args.output_dir, "fewclue_submit_examples_" + time_stamp)
        save_fewclue_prediction(predict_path, data_args.task_name, pred_ret, verbalizer, ids_to_labels)
        # 这段代码片段用于进行模型的预测并保存预测结果。
        #
        # 首先，判断是否需要进行预测（do_predict）以及是否存在测试数据集（test_ds）。如果条件满足，就会执行以下操作：
        #
        # 进行预测：通过调用trainer对象的predict方法对测试数据集（test_ds）进行预测。预测结果保存在pred_ret变量中。
        #
        # 记录日志信息：通过调用logger.info方法记录一条日志信息，表示预测已完成。
        #
        # 设置预测结果保存路径：根据训练参数中的输出目录（output_dir）和时间戳（time_stamp），构建预测结果保存路径（predict_path）。
        #
        # 保存预测结果：通过调用save_fewclue_prediction函数，将预测结果（pred_ret）、任务名称（data_args.task_name）、verbalizer和ids_to_labels等信息保存到指定的路径（predict_path）中。
        #
        # 这段代码的作用是在进行模型预测时，通过调用trainer对象的predict方法获取预测结果，并将预测结果保存到指定路径中，以便后续分析和提交预测结果。

    # Label unsupervised data.
    if data_args.do_label and unlabeled_ds is not None:
        label_ret = trainer.predict(unlabeled_ds)
        logger.info("Labeling done.")
        pseudo_path = os.path.join(training_args.output_dir, "pseudo_data_" + time_stamp + ".txt")
        save_pseudo_data(pseudo_path, data_args.task_name, label_ret, verbalizer, ids_to_labels)
        # 这段代码片段用于对未标记的数据进行标记（labeling）并保存标记结果。
        #
        # 首先，判断是否需要进行数据标记（do_label）以及是否存在未标记的数据集（unlabeled_ds）。如果条件满足，就会执行以下操作：
        #
        # 进行数据标记：通过调用trainer对象的predict方法对未标记的数据集（unlabeled_ds）进行数据标记。标记结果保存在label_ret变量中。
        #
        # 记录日志信息：通过调用logger.info方法记录一条日志信息，表示数据标记已完成。
        #
        # 设置标记结果保存路径：根据训练参数中的输出目录（output_dir）和时间戳（time_stamp），构建标记结果保存路径（pseudo_path）。
        #
        # 保存标记结果：通过调用save_pseudo_data函数，将标记结果（label_ret）、任务名称（data_args.task_name）、verbalizer和ids_to_labels等信息保存到指定的路径（pseudo_path）中。
        #
        # 这段代码的作用是在需要对未标记的数据进行标记时，通过调用trainer对象的predict方法获取标记结果，并将标记结果保存到指定路径中，以便后续分析和使用标记数据进行训练或其他用途。

    # Export static model.
    if training_args.do_export:
        template = prompt_model.template
        template_keywords = template.extract_template_keywords(template.prompt)
        input_spec = [
            InputSpec(shape=[None, None], dtype="int64"),  # input_ids,
            InputSpec(shape=[None, None], dtype="int64"),  # token_type_ids
            InputSpec(shape=[None, None], dtype="int64"),  # position_ids
            InputSpec(shape=[None, None, None, None], dtype="float32"),  # attention_mask
            InputSpec(shape=[None], dtype="int64"),  # masked_positions
            InputSpec(shape=[None, None], dtype="int64"),  # soft_token_ids
        ]
        if "encoder" in template_keywords:
            input_spec.append(InputSpec(shape=[None, None], dtype="int64"))  # encoder_ids
        export_path = os.path.join(training_args.output_dir, "export")
        trainer.export_model(export_path, input_spec=input_spec, export_type=model_args.export_type)

    # 这段代码片段用于导出模型。
    #
    # 首先，判断是否需要进行模型导出（do_export）。如果条件满足，就会执行以下操作：
    #
    # 获取模板和模板关键词：通过访问prompt_model对象的template属性，获取模板对象（template）。然后使用模板对象的extract_template_keywords方法提取模板中的关键词（template_keywords）。
    #
    # 定义输入规范（InputSpec）：根据模型的输入要求，定义输入规范（input_spec）。其中包括input_ids、token_type_ids、position_ids、attention_mask、masked_positions和soft_token_ids等输入。
    #
    # 如果模板关键词中包含"encoder"，则添加额外的输入规范：根据模板关键词中是否包含"encoder"关键词，决定是否需要添加额外的输入规范encoder_ids。
    #
    # 设置导出路径：根据训练参数中的输出目录（output_dir）和固定的导出目录名称（"export"），构建导出路径（export_path）。
    #
    # 导出模型：通过调用trainer对象的export_model方法，将模型导出到指定路径（export_path）。同时，传递输入规范（input_spec）和导出类型（export_type）参数。
    #
    # 这段代码的作用是在需要导出模型时，根据模型的输入要求和模板关键词，定义输入规范并将模型导出到指定路径中，以便后续使用或部署模型。

if __name__ == "__main__":
    main()
