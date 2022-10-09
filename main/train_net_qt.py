import matplotlib

matplotlib.use('Agg')

import logging
import os
from collections import OrderedDict

import torch
import socket
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    SemSegEvaluator,
)
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
import datetime
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    verify_results,
)
from main.train_net_mf import MFTrainer
# MaskFormer
from mask_former import (
    DETRPanopticDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    WeakShotSemSegMapper,
    add_mask_former_config,
)
from qt_former.evaluation import WeakShotSemSegEvaluator
from mask_former.utils.viz import c_print_csv_format
from qt_former import add_qt_former_config


class QTTrainer(MFTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ['weakshot_sem_seg']:
            evaluator_list += [WeakShotSemSegEvaluator(dataset_name, eval_type='full',
                                                       distributed=True, output_dir=output_folder),
                               WeakShotSemSegEvaluator(dataset_name, eval_type='base',
                                                       distributed=True, output_dir=output_folder),
                               WeakShotSemSegEvaluator(dataset_name, eval_type='novel',
                                                       distributed=True, output_dir=output_folder),
                               ]

        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type in [
            "coco_panoptic_seg",
            "ade20k_panoptic_seg",
            "cityscapes_panoptic_seg",
        ]:
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                    torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                    torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        if evaluator_type == "cityscapes_panoptic_seg":
            assert (
                    torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):

        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
            mapper = MaskFormerPanopticDatasetMapper(cfg, True)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "detr_panoptic":
            mapper = DETRPanopticDatasetMapper(cfg, True)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "weakshot_sem_seg_mapper":
            mapper = WeakShotSemSegMapper(cfg, True)
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
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                c_print_csv_format(results_i, logger)
                logger.info(cfg.OUTPUT_DIR)

        if len(results) == 1:
            results = list(results.values())[0]
        return results


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_mask_former_config(cfg)
    add_qt_former_config(cfg)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    mk_out_dir(cfg)

    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask_former")
    return cfg


def mk_out_dir(cfg):
    cfg.OUTPUT_DIR += f"/Exp{cfg.OUTPUT_PREFIX}"

    cfg.OUTPUT_DIR += f'_Q{cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES}'

    if cfg.MODEL.MASK_FORMER.FIXED_MATCHER:
        cfg.OUTPUT_DIR += '_MFix'

    if cfg.MODEL.MASK_FORMER.FREEZE_QUERY:
        cfg.OUTPUT_DIR += '_QFreeze'

    cfg.OUTPUT_DIR += f'_Q{cfg.MODEL.MASK_FORMER.TRANS_QUERY}'

    if cfg.MODEL.MASK_FORMER.MAKE_GSIM:
        cfg.OUTPUT_DIR += f'_GSIM{cfg.MODEL.MASK_FORMER.GSIM_WEIGHT}'

    cfg.OUTPUT_DIR += f"_G{torch.cuda.device_count()}{socket.gethostname()[:10]}" \
                      f"_{datetime.datetime.now().strftime('%m%d%H%M')}"
    return


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = QTTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )

        res = QTTrainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(QTTrainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = QTTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    # print_pc(
    #     {
    #         'model': trainer.model,
    #         'backbone': trainer.model.backbone,
    #         'seghead': trainer.model.sem_seg_head,
    #     }
    # )
    return trainer.train()


def print_pc(module_dict, printf=print):
    for name, module in module_dict.items():
        total_params = sum(p.numel() for p in module.parameters())
        total_trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)

        printf(f'{total_trainable_params / 1e6:.1f}M/{total_params / 1e6:.1f}M training/total params in {name}.')
    return


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
