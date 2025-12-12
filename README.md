# DeepLearning\_Model\_optimization\_Assignment\_3



1. **run training script train.py**



With WandB :

python train.py --batch\_size 64 --epochs 150 --lr 0.1 --wandb



or without wandb:



python train.py --batch\_size 64 --epochs 150 --lr 0.1



=> best\_mobilenetv2\_cifar10.pth will be saved after training.



**2. Notes on quantization code**



=> quant\_utils.py contains full implementation of:



&nbsp;	uniform quantization



&nbsp;	per-channel weight quant



&nbsp;	activation quant



&nbsp;	packing (int4/int2)



&nbsp;	metadata tracking



=> compress\_and\_eval.py contains pipeline to:



&nbsp;	load your best\_mobilenetv2\_cifar10.pth



&nbsp;	quantize weights + activations



&nbsp;	simulate/fake-quant finetune



&nbsp;	compute model sizes, activation sizes, ratios



&nbsp;	WandB logging



&nbsp;	save compressed model





compress\_and\_eval.py uses the quant\_utils.py . It runs quantization over weights, optionally finetunes with fake-quant 

(by applying dequantized weights and training), measures activation sizes, computes storage and compression ratios, 

logs to WandB, and saves quantized metadata.



**3. Quantize without finetuning**



&nbsp;  python compress\_and\_eval.py --checkpoint best\_mobilenetv2\_cifar10.pth \\

&nbsp;    --weight-bits 4 --act-bits 8 --batch-size 64

&nbsp;  

&nbsp;  Quantize + Finetune 3–5 epochs :

&nbsp;

&nbsp;  python compress\_and\_eval.py --checkpoint best\_mobilenetv2\_cifar10.pth \\

&nbsp;   --weight-bits 4 --act-bits 6 --finetune-epochs 5 --finetune-lr 1e-3 \\

&nbsp;   --batch-size 32

&nbsp;  

&nbsp;  Enable WandB logging :

&nbsp;  

&nbsp;  python compress\_and\_eval.py --checkpoint best\_mobilenetv2\_cifar10.pth \\

&nbsp;   --weight-bits 4 --act-bits 8 --wandb



**4. what to expect after running compress\_and\_eval.py**



The script prints:



Baseline test accuracy



Quantized model storage (bytes + MB)



Activation memory (original vs quantized)



Compression ratios



Post-quantization accuracy (dequantized model)



Optional finetuning accuracy and saves quantized\_model\_w4\_a8.pth



**5. Plot Loss \& Accuracy Curves (use training\_log.csv)**

&nbsp;  python plot\_training\_curves.py



**6. Collect Quantization Experiment Results ()**

&nbsp;  python compress\_and\_eval.py --weight-bits 8 --act-bits 8

&nbsp;  python compress\_and\_eval.py --weight-bits 6 --act-bits 8

&nbsp;  python compress\_and\_eval.py --weight-bits 4 --act-bits 8

&nbsp;  python compress\_and\_eval.py --weight-bits 2 --act-bits 8



**7. Plot Model Size vs Bitwidth \& Accuracy (use quantization\_results.csv)**

&nbsp;  python plot\_quant\_results.py



**8. Plot on WandB**

python compress\_and\_eval.py --weight-bits 8 --act-bits 8 --wandb

python compress\_and\_eval.py --weight-bits 6 --act-bits 8 --wandb

python compress\_and\_eval.py --weight-bits 4 --act-bits 6 --wandb

python compress\_and\_eval.py --weight-bits 4 --act-bits 4 --wandb

python compress\_and\_eval.py --weight-bits 2 --act-bits 8 --wandb







