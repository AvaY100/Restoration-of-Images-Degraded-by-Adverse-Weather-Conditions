import os, sys, pdb
import json
from io import StringIO 
import warnings
warnings.filterwarnings("ignore")
import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def parse_result_list(result_output):
    result_output = result_output[5:]
    result_output = result_output[:6]
    result_names = [ "AP", "AP50", "AP75", "AP(small)", "AP(medium)", "AP(large)" ]
    result_output = [ float(l.split("=")[-1].strip()) for l in result_output ]
    return dict(zip(result_names, result_output))


class StdOutCapturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout


class StdErrCapturing(list):
    def __enter__(self):
        self._stderr = sys.stderr
        sys.stderr = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stderr = self._stderr


def build_parser(parser):
    parser.add_argument("--model_preds", type=str, required=True)
    parser.add_argument("--gt_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    return parser


if __name__=="__main__":
    args = build_parser(argparse.ArgumentParser()).parse_args()
    # if os.path.exists(args.output_file):
    #     print("Already evaluated, exiting")
    #     sys.exit()
    
    with open(args.gt_file, "r") as f:
        dataset = json.load(f)
    print(dataset["categories"])

    OUT = {}
    coco_gt = COCO(args.gt_file)
    coco_dt = coco_gt.loadRes(args.model_preds)
    # pdb.set_trace()
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    # coco_eval.params.catIds = [
    #     1, 2, 3, 4, 6, 8,
    # ]
    coco_eval.params.catIds = [
        1, 2, 3, 8,
    ]

    with open(args.gt_file, "r") as f:
        dataset = json.load(f)
    
    with StdOutCapturing() as result_output:
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    coco_output = parse_result_list(result_output)
    OUT["all"] = coco_output
    
    # compute the per-class mAP
    cats = dataset["categories"]
    for cat_info in cats:
        cat_id = cat_info["id"]
        cat_name = cat_info["name"]
        with StdOutCapturing() as result_output:
            coco_eval.params.catIds = [cat_id]
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
        coco_output = parse_result_list(result_output)
        if coco_output["AP"] == -1:
            continue
        OUT[cat_name] = coco_output
        
    print(OUT['all'])

    # save the results
    with open(args.output_file, "w") as f:
        json.dump(OUT, f, indent=4)