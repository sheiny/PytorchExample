//source https://github.com/Wizaron/pytorch-cpp-inference

#include "infer.h"

int main(int argc, char **argv) {

  int image_height = 224;
  int image_width = 224;

  // Read labels
  std::vector<std::string> labels = {"Ant", "Bee"};

  std::vector<double> mean = {0.485, 0.456, 0.406};
  std::vector<double> std = {0.229, 0.224, 0.225};

  // Examples
  //cv::Mat image = cv::imread("../hymenoptera_data/val/bees/2060668999_e11edb10d0.jpg");
  cv::Mat image = cv::imread("../hymenoptera_data/val/bees/2104135106_a65eede1de.jpg");
  //cv::Mat image = cv::imread("../hymenoptera_data/val/ants/1440002809_b268d9a66a.jpg");
  
  std::shared_ptr<torch::jit::script::Module> model = std::make_shared<torch::jit::script::Module>(torch::jit::load("../traced_resnet_model.pt"));

  std::string pred, prob;
  tie(pred, prob) = infer(image, image_height, image_width, mean, std, labels, model);

  std::cout << "PREDICTION  : " << pred << std::endl;
  std::cout << "CONFIDENCE  : " << prob << std::endl;

  imshow( "Evaluation Image", image ); 
  std::cout<<"press any key to continue"<<std::endl;
  cv::waitKey(0);
  return 0;
}
