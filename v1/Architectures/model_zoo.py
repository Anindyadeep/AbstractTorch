try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import warnings
    warnings.filterwarnings("ignore")
    print("All modules installed successfully ...")
except ModuleNotFoundError as e:
    print(f"ERROR: {e} Install all the modules properly ...")


class ModelZoo(nn.Module):
    def __init__(self, model, pretrained, num_classes):
        super(ModelZoo, self).__init__()
        self.num_classes = num_classes
        self.model = model
        self.pretrained = pretrained
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    def _set_requires_grad(self, model, extracting_grads=True):
        if extracting_grads:
            for param in model.parameters():
                param.requires_grad = False

    def create_model(self):
        if 'resnet18' == self.model:
            print('Using ResNet18 ...')
            model = models.resnet18(pretrained=self.self.pretrained)
            if self.pretrained:
                self._set_requires_grad(model)
            else:
                self._set_requires_grad(model, extracting_grads=False)
            model.fc = nn.Linear(model.fc.in_features, self.self.num_classes)

        elif 'resnet34' == self.model:
            print('Using ResNet34 ...')
            model = models.resnet34(pretrained=self.pretrained)
            if self.pretrained:
                self._set_requires_grad(model)
            else:
                self._set_requires_grad(model, extracting_grads=False)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)

        elif 'resnet50' == self.model:
            print('Using ResNet50 ...')
            model = models.resnet50(pretrained=self.pretrained)
            if self.pretrained:
                self._set_requires_grad(model)
            else:
                self._set_requires_grad(model, extracting_grads=False)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)

        elif 'resnet101' == self.model:
            print('Using ResNet101 ...')
            model = models.resnet101(pretrained=self.pretrained)
            if self.pretrained:
                self._set_requires_grad(model)
            else:
                self._set_requires_grad(model, extracting_grads=False)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)

        elif 'resnet152' == self.model:
            print('Using ResNet512 ...')
            model = models.resnet152(pretrained=self.pretrained)
            if self.pretrained:
                self._set_requires_grad(model)
            else:
                self._set_requires_grad(model, extracting_grads=False)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)

        elif 'alexnet' == self.model:
            print('Using AlexNet ...')
            model = models.alexnet(pretrained=self.pretrained)
            if self.pretrained:
                self._set_requires_grad(model)
            else:
                self._set_requires_grad(model, extracting_grads=False)
            model.classifier[6] = nn.Linear(4096, self.num_classes)

        elif 'vgg11' == self.model:
            print('Using VGG11 ...')
            model = models.vgg11(pretrained=self.pretrained)
            if self.pretrained:
                self._set_requires_grad(model)
            else:
                self._set_requires_grad(model, extracting_grads=False)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, self.num_classes)

        elif 'vgg11_bn' == self.model:
            print('Using VGG11 (BN)...')
            model = models.vgg11_bn(pretrained=self.pretrained)
            if self.pretrained:
                self._set_requires_grad(model)
            else:
                self._set_requires_grad(model, extracting_grads=False)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, self.num_classes)

        elif 'vgg13' == self.model:
            print('Using VGG13 ...')
            model = models.vgg13(pretrained=self.pretrained)
            if self.pretrained:
                self._set_requires_grad(model)
            else:
                self._set_requires_grad(model, extracting_grads=False)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, self.num_classes)

        elif 'vgg13_bn' == self.model:
            print('Using VGG13 (BN) ...')
            model = models.vgg13_bn(pretrained=self.pretrained)
            if self.pretrained:
                self._set_requires_grad(model)
            else:
                self._set_requires_grad(model, extracting_grads=False)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, self.num_classes)

        elif 'vgg16' == self.model:
            print('Using VGG16 ...')
            model = models.vgg16(pretrained=self.pretrained)
            if self.pretrained:
                self._set_requires_grad(model)
            else:
                self._set_requires_grad(model, extracting_grads=False)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, self.num_classes)

        elif 'vgg16_bn' == self.model:
            print('Using VGG16 (BN) ...')
            model = models.vgg16_bn(pretrained=self.pretrained)
            if self.pretrained:
                self._set_requires_grad(model)
            else:
                self._set_requires_grad(model, extracting_grads=False)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, self.num_classes)

        elif 'vgg19' == self.model:
            print('Using VGG19 ...')
            model = models.vgg19(pretrained=self.pretrained)
            if self.pretrained:
                self._set_requires_grad(model)
            else:
                self._set_requires_grad(model, extracting_grads=False)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, self.num_classes)

        elif 'vgg19_bn' == self.model:
            print('Using VGG19 (BN) ...')
            model = models.vgg19_bn(pretrained=self.pretrained)
            if self.pretrained:
                self._set_requires_grad(model)
            else:
                self._set_requires_grad(model, extracting_grads=False)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, self.num_classes)

        elif 'squeezenet1_0' == self.model:
            print('Using SqueezeNet v 1.0 ...')
            model = models.squeezenet1_0(pretrained=self.pretrained)
            if self.pretrained:
                self._set_requires_grad(model)
            else:
                self._set_requires_grad(model, extracting_grads=False)
            model.classifier[1] = nn.Conv2d(512, self.num_classes, kernel_size=(1, 1), stride=(1, 1))
            model.num_classes = self.num_classes

        elif 'squeezenet1_1' == self.model:
            print('Using SqueezeNet v 1.1 ...')
            model = models.squeezenet1_1(pretrained=self.pretrained)
            if self.pretrained:
                self._set_requires_grad(model)
            else:
                self._set_requires_grad(model, extracting_grads=False)
            model.classifier[1] = nn.Conv2d(512, self.num_classes, kernel_size=(1, 1), stride=(1, 1))
            model.num_classes = self.num_classes

        elif 'densenet121' == self.model:
            print('Using DenseNet121 ...')
            model = models.densenet121(pretrained=self.pretrained)
            if self.pretrained:
                self._set_requires_grad(model)
            else:
                self._set_requires_grad(model, extracting_grads=False)
            model.classifier = nn.Linear(model.classifier.in_features, self.num_classes)

        elif 'densenet161' == self.model:
            print('Using DenseNet161 ...')
            model = models.densenet161(pretrained=self.pretrained)
            if self.pretrained:
                self._set_requires_grad(model)
            else:
                self._set_requires_grad(model, extracting_grads=False)
            model.classifier = nn.Linear(model.classifier.in_features, self.num_classes)

        elif 'densenet169' == self.model:
            print('Using DenseNet169 ...')
            model = models.densenet169(pretrained=self.pretrained)
            if self.pretrained:
                self._set_requires_grad(model)
            else:
                self._set_requires_grad(model, extracting_grads=False)
            model.classifier = nn.Linear(model.classifier.in_features, self.num_classes)

        elif 'densenet201' == self.model:
            print('Using DenseNet201 ...')
            model = models.densenet201(pretrained=self.pretrained)
            if self.pretrained:
                self._set_requires_grad(model)
            else:
                self._set_requires_grad(model, extracting_grads=False)
            model.classifier = nn.Linear(model.classifier.in_features, self.num_classes)

        elif 'googlenet' == self.model:
            print('Using GoogleNet ...')
            model = models.googlenet(pretrained=self.pretrained)
            if self.pretrained:
                self._set_requires_grad(model)
            else:
                self._set_requires_grad(model, extracting_grads=False)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)

        elif 'shufflenet_v2_x0_5' == self.model:
            print('Using ShuffleNet v2 0.5 ...')
            model = models.shufflenet_v2_x0_5(pretrained=self.pretrained)
            if self.pretrained:
                self._set_requires_grad(model)
            else:
                self._set_requires_grad(model, extracting_grads=False)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)

        elif 'shufflenet_v2_x1_0' == self.model:
            print('Using ShuffleNet v2 1.0 ...')
            model = models.shufflenet_v2_x1_0(pretrained=self.pretrained)
            if self.pretrained:
                self._set_requires_grad(model)
            else:
                self._set_requires_grad(model, extracting_grads=False)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)

        elif 'mobilenet_v2' == self.model:
            print('Using MobileNet v2 ...')
            model = models.mobilenet_v2(pretrained=self.pretrained)
            if self.pretrained:
                self._set_requires_grad(model)
            else:
                self._set_requires_grad(model, extracting_grads=False)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)

        elif 'resnext50_32x4d' == self.model:
            print('Using ResNeXt50 32x4d ...')
            model = models.resnext50_32x4d(pretrained=self.pretrained)
            if self.pretrained:
                self._set_requires_grad(model)
            else:
                self._set_requires_grad(model, extracting_grads=False)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)

        elif 'resnext101_32x8d' == self.model:
            print('Using ResNeXt101 32x8d ...')
            model = models.resnext101_32x8d(pretrained=self.pretrained)
            if self.pretrained:
                self._set_requires_grad(model)
            else:
                self._set_requires_grad(model, extracting_grads=False)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)

        elif 'wide_resnet50_2' == self.model:
            print('Using WideResNet 50_2 ...')
            model = models.wide_resnet50_2(pretrained=self.pretrained)
            if self.pretrained:
                self._set_requires_grad(model)
            else:
                self._set_requires_grad(model, extracting_grads=False)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)

        elif 'wide_resnet101_2' == self.model:
            print('Using WideResNet 101_2 ...')
            model = models.wide_resnet101_2(pretrained=self.pretrained)
            if self.pretrained:
                self._set_requires_grad(model)
            else:
                self._set_requires_grad(model, extracting_grads=False)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)

        elif 'mnasnet0_5' == self.model:
            print('Using MnasNet v 0.5 ...')
            model = models.mnasnet0_5(pretrained=self.pretrained)
            if self.pretrained:
                self._set_requires_grad(model)
            else:
                self._set_requires_grad(model, extracting_grads=False)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)

        elif 'mnasnet1_0' == self.model:
            print('Using MnasNet v 1.0 ...')
            model = models.mnasnet1_0(pretrained=self.pretrained)
            if self.pretrained:
                self._set_requires_grad(model)
            else:
                self._set_requires_grad(model, extracting_grads=False)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)

        else:
            print('Using Default Inception v3 ...')
            model = models.inception_v3(pretrained=self.pretrained)
            if self.pretrained:
                self._set_requires_grad(model)
            else:
                self._set_requires_grad(model, extracting_grads=False)
            model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, self.num_classes)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        return model.to(self.device)


