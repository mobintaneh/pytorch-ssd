from torch import nn

import modules


class SSD(nn.Module):
    def __init__(self, backbone, num_classes):
        super(SSD, self).__init__()

        self.num_classes = num_classes
        self.backbone = backbone

        self.extras = nn.ModuleList([
            modules.InvertedResidual(1280, 512, stride=2, expand_ratio=0.2),
            modules.InvertedResidual(512, 256, stride=2, expand_ratio=0.25),
            modules.InvertedResidual(256, 256, stride=2, expand_ratio=0.5),
            modules.InvertedResidual(256, 64, stride=2, expand_ratio=0.25)
        ])

        self.classification_headers = nn.ModuleList([
            modules.SeparableConv2d(in_channels=round(576), out_channels=6 * 4, kernel_size=3, padding=1),
            modules.SeparableConv2d(in_channels=1280, out_channels=6 * 4, kernel_size=3, padding=1),
            modules.SeparableConv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1),
            modules.SeparableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
            modules.SeparableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
            modules.SeparableConv2d(in_channels=64, out_channels=6 * 4, kernel_size=1)
        ])

        self.regression_headers = nn.ModuleList([
            modules.SeparableConv2d(in_channels=round(576), out_channels=6 * num_classes, kernel_size=3, padding=1),
            modules.SeparableConv2d(in_channels=1280, out_channels=6 * num_classes, kernel_size=3, padding=1),
            modules.SeparableConv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
            modules.SeparableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
            modules.SeparableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=6 * num_classes, kernel_size=1),
        ])

    def forward(self, x):
        confidences = []
        locations = []
        start_layer_index = 0
        header_index = 0

        for layer in self.backbone:
            x = layer(x)

        for layer in self.extras:
            x = layer(x)
            confidence, location = self.compute_header(header_index, x)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)

        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)

        if self.is_test:
            confidences = F.softmax(confidences, dim=2)
            boxes = box_utils.convert_locations_to_boxes(
                locations, self.priors, self.config.center_variance, self.config.size_variance
            )
            boxes = box_utils.center_form_to_corner_form(boxes)
            return confidences, boxes
        else:
            return confidences, locations

    def compute_header(self, i, x):
        confidence = self.classification_headers[i](x)
        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        confidence = confidence.view(confidence.size(0), -1, self.num_classes)

        location = self.regression_headers[i](x)
        location = location.permute(0, 2, 3, 1).contiguous()
        location = location.view(location.size(0), -1, 4)

        return confidence, location

    def init_from_base_net(self, model):
        self.base_net.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage), strict=True)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def init_from_pretrained_ssd(self, model):
        state_dict = torch.load(model, map_location=lambda storage, loc: storage)
        state_dict = {k: v for k, v in state_dict.items() if
                      not (k.startswith("classification_headers") or k.startswith("regression_headers"))}
        model_dict = self.state_dict()
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def init(self):
        self.base_net.apply(_xavier_init_)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def load(self, model):
        self.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)


def ssdlite_mobinenetv2():
    pass