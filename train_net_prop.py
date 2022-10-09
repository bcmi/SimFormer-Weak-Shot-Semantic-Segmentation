import matplotlib
import sys

matplotlib.use('Agg')
sys.path.append('.')
import torch
import datetime
import logging
import os
from collections import OrderedDict
import socket
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    DatasetEvaluators,
)
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    verify_results,
)

from main.train_net_mf import MFTrainer
from mask_former import add_mask_former_config
from mask_former import SemanticSegmentorWithTTA

from prop_former.evaluation import WeakShotSemSegEvaluator
from prop_former import add_prop_former_config
from prop_former import WeakShotMapperTraining
from prop_former.shared import c_print_csv_format


class PropTrainer(MFTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ['weakshot_sem_seg']:
            evaluator_list += [
                WeakShotSemSegEvaluator(
                    cfg, dataset_name, eval_bias=eval_bias, eval_type='full',
                    distributed=True, output_dir=output_folder
                ) for eval_bias in cfg.EVAL.BIAS
            ]
        # if cfg.LOSS.BASE_REGION_SEG != 0:
        #     seenbias_e = WeakShotSemSegSeenBiasEvaluator(cfg, dataset_name, eval_bias=None,
        #                                                  eval_type='full', distributed=True, output_dir=output_folder)
        #     evaluator_list.append(seenbias_e)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type))
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.INPUT.DATASET_MAPPER_NAME == "weakshot_sem_seg_mapper":
            mapper = WeakShotMapperTraining(cfg, True)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "pair_mapper":
            from prop_former.data.dataset_mappers.weakshot_mapper_training_pair import WeakShotMapperTrainingPair
            mapper = WeakShotMapperTrainingPair(cfg, True)
        else:
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        logger = logging.getLogger('detectron2')
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )
        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method.")
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(results_i, dict), \
                    "Evaluator must return a dict on the main process. Got {} instead.".format(results_i)
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                c_print_csv_format(results_i, logger)
                logger.info(f'{cfg.MODEL.WEIGHTS}')
                logger.info(cfg.OUTPUT_DIR)

        if len(results) == 1:
            results = list(results.values())[0]
        return results

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_mask_former_config(cfg)
    add_prop_former_config(cfg)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    mk_out_dir(cfg)

    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask_former")
    return cfg


def mk_out_dir(cfg):
    ostr = f"/{cfg.OUTPUT_PREFIX}"
    ostr += f"_G{torch.cuda.device_count()}{socket.gethostname()[:5]}" \
            f"_{datetime.datetime.now().strftime('%m%d%H%M')}"

    cfg.OUTPUT_DIR += ostr
    return


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = PropTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        if cfg.TEST.AUG.ENABLED:
            res = PropTrainer.test_with_TTA(cfg, model)
            print('vvvvvvvvvvvvvvvvvvvvvvvvv')
            print(res)
            print('^^^^^^^^^^^^^^^^^^^^^^^^^')

            logger = logging.getLogger('detectron2')
            logger.info('vvvvvvvvvvvvvvvvvvvvvvvvv')
            logger.info(str(res))
            logger.info('^^^^^^^^^^^^^^^^^^^^^^^^^')
        else:
            res = PropTrainer.test(cfg, model)

        if comm.is_main_process():
            verify_results(cfg, res)

        return res

    trainer = PropTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    logger = logging.getLogger('detectron2')
    logger.info(f'\n\n@@@@@@@@@@@\n{cfg}\n@@@@@@@@@@@@\n\n')
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    args.num_gpus = torch.cuda.device_count()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
