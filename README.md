# CNN-Components-Back-Propagation

## Model Requirements Checks

| Requirement | Status |
|------------|---------|
| Parameters < 20k | ![Test Status](https://github.com/kalekarnn/CNN-Components-Back-Propagation/actions/workflows/pipeline.yml/badge.svg?event=push&label=parameters) |
| Batch Normalization | ![Test Status](https://github.com/kalekarnn/CNN-Components-Back-Propagation/actions/workflows/pipeline.yml/badge.svg?event=push&label=batch-norm) |
| Dropout | ![Test Status](https://github.com/kalekarnn/CNN-Components-Back-Propagation/actions/workflows/pipeline.yml/badge.svg?event=push&label=dropout) |
| GAP/FC Layer | ![Test Status](https://github.com/kalekarnn/CNN-Components-Back-Propagation/actions/workflows/pipeline.yml/badge.svg?event=push&label=architecture) |
| Accuracy > 99.4% | ![Test Status](https://github.com/kalekarnn/CNN-Components-Back-Propagation/actions/workflows/pipeline.yml/badge.svg?event=push&label=accuracy) |


## Model Architecture

```
        self.conv1 = nn.Sequential(
          nn.Conv2d(1, 32, 5, padding=1),
          nn.ReLU(),
          nn.BatchNorm2d(32),
          nn.Dropout(0.05),
          nn.Conv2d(32, 24, 3, stride=1, padding=1),
          nn.ReLU(),
          nn.BatchNorm2d(24),
          nn.Dropout(0.1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(24, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 24, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.Dropout(0.1)
        )

        self.conv4 = nn.Sequential(
              nn.Conv2d(24,16,3,stride=1,padding=1),
              nn.ReLU(),
              nn.BatchNorm2d(16)
          )


        self.conv5 = nn.Sequential(
            nn.Conv2d(16, 10, 3, stride=1)
        )

        self.gap = nn.AvgPool2d(5)
```

## Model Summary 

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 26, 26]             832
              ReLU-2           [-1, 32, 26, 26]               0
       BatchNorm2d-3           [-1, 32, 26, 26]              64
           Dropout-4           [-1, 32, 26, 26]               0
            Conv2d-5           [-1, 24, 26, 26]           6,936
              ReLU-6           [-1, 24, 26, 26]               0
       BatchNorm2d-7           [-1, 24, 26, 26]              48
           Dropout-8           [-1, 24, 26, 26]               0
            Conv2d-9           [-1, 16, 13, 13]           3,472
             ReLU-10           [-1, 16, 13, 13]               0
      BatchNorm2d-11           [-1, 16, 13, 13]              32
          Dropout-12           [-1, 16, 13, 13]               0
           Conv2d-13             [-1, 24, 7, 7]           3,480
             ReLU-14             [-1, 24, 7, 7]               0
      BatchNorm2d-15             [-1, 24, 7, 7]              48
          Dropout-16             [-1, 24, 7, 7]               0
           Conv2d-17             [-1, 16, 7, 7]           3,472
             ReLU-18             [-1, 16, 7, 7]               0
      BatchNorm2d-19             [-1, 16, 7, 7]              32
           Conv2d-20             [-1, 10, 5, 5]           1,450
        AvgPool2d-21             [-1, 10, 1, 1]               0
================================================================
Total params: 19,866
Trainable params: 19,866
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 1.29
Params size (MB): 0.08
Estimated Total Size (MB): 1.37
----------------------------------------------------------------

```

## Training Logs

```
loss=0.11945220082998276 batch_id=468: 100%|██████████| 469/469 [02:13<00:00,  3.51it/s]
Test set: Average loss: 0.0767, Accuracy: 9784/10000 (97.84%)

loss=0.17083798348903656 batch_id=468: 100%|██████████| 469/469 [02:16<00:00,  3.45it/s]
Test set: Average loss: 0.0476, Accuracy: 9857/10000 (98.57%)

loss=0.060158368200063705 batch_id=468: 100%|██████████| 469/469 [02:12<00:00,  3.53it/s]
Test set: Average loss: 0.0416, Accuracy: 9876/10000 (98.76%)

loss=0.015993479639291763 batch_id=468: 100%|██████████| 469/469 [02:13<00:00,  3.53it/s]
Test set: Average loss: 0.0369, Accuracy: 9885/10000 (98.85%)

loss=0.04363897442817688 batch_id=468: 100%|██████████| 469/469 [02:12<00:00,  3.54it/s]
Test set: Average loss: 0.0285, Accuracy: 9916/10000 (99.16%)

loss=0.0063562337309122086 batch_id=468: 100%|██████████| 469/469 [02:13<00:00,  3.52it/s]
Test set: Average loss: 0.0300, Accuracy: 9897/10000 (98.97%)

loss=0.03154856711626053 batch_id=468: 100%|██████████| 469/469 [02:12<00:00,  3.55it/s]
Test set: Average loss: 0.0295, Accuracy: 9899/10000 (98.99%)

loss=0.021261846646666527 batch_id=468: 100%|██████████| 469/469 [02:12<00:00,  3.53it/s]
Test set: Average loss: 0.0239, Accuracy: 9925/10000 (99.25%)

loss=0.04127318412065506 batch_id=468: 100%|██████████| 469/469 [02:12<00:00,  3.54it/s]
Test set: Average loss: 0.0205, Accuracy: 9931/10000 (99.31%)

loss=0.04609071835875511 batch_id=468: 100%|██████████| 469/469 [02:12<00:00,  3.54it/s]
Test set: Average loss: 0.0225, Accuracy: 9920/10000 (99.20%)

loss=0.07690980285406113 batch_id=468: 100%|██████████| 469/469 [02:11<00:00,  3.57it/s]
Test set: Average loss: 0.0212, Accuracy: 9940/10000 (99.40%)

loss=0.03543000668287277 batch_id=468: 100%|██████████| 469/469 [02:11<00:00,  3.56it/s]
Test set: Average loss: 0.0199, Accuracy: 9945/10000 (99.45%)

loss=0.018456505611538887 batch_id=468: 100%|██████████| 469/469 [02:11<00:00,  3.55it/s]
Test set: Average loss: 0.0188, Accuracy: 9940/10000 (99.40%)

loss=0.003524158848449588 batch_id=468: 100%|██████████| 469/469 [02:12<00:00,  3.54it/s]
Test set: Average loss: 0.0181, Accuracy: 9940/10000 (99.40%)

loss=0.017058631405234337 batch_id=468: 100%|██████████| 469/469 [02:11<00:00,  3.55it/s]
Test set: Average loss: 0.0177, Accuracy: 9950/10000 (99.50%)

loss=0.0624438114464283 batch_id=468: 100%|██████████| 469/469 [02:12<00:00,  3.53it/s]
Test set: Average loss: 0.0201, Accuracy: 9944/10000 (99.44%)

loss=0.06440434604883194 batch_id=468: 100%|██████████| 469/469 [02:11<00:00,  3.57it/s]
Test set: Average loss: 0.0192, Accuracy: 9943/10000 (99.43%)

loss=0.0804075300693512 batch_id=468: 100%|██████████| 469/469 [02:12<00:00,  3.55it/s]
Test set: Average loss: 0.0180, Accuracy: 9948/10000 (99.48%)

loss=0.0019424442434683442 batch_id=468: 100%|██████████| 469/469 [02:12<00:00,  3.53it/s]
Test set: Average loss: 0.0177, Accuracy: 9947/10000 (99.47%)

loss=0.044174712151288986 batch_id=468: 100%|██████████| 469/469 [02:12<00:00,  3.55it/s]
Test set: Average loss: 0.0172, Accuracy: 9942/10000 (99.42%)

```
