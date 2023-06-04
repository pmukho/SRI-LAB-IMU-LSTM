#include <torch/torch.h>
#include <iostream>
#include <vector>

class LSTMImpl : public torch::nn::Module {
  private:
    int64_t input_size;
    int64_t hidden_size;
    int64_t num_layers;
    int64_t output_size;
    torch::nn::LSTM lstm;
    // torch::nn::Linear fc;

  public:
    LSTMImpl(int64_t input_size, int64_t hidden_size, int64_t num_layers, int64_t output_size) 
    : input_size(input_size), hidden_size(hidden_size), num_layers(num_layers), output_size(output_size), 
    lstm(torch::nn::LSTM(torch::nn::LSTMOptions(input_size, hidden_size).num_layers(num_layers))),
    fc(torch::nn::Linear(hidden_size, output_size)) {
      // lstm = torch::nn::LSTM(torch::nn::LSTMOptions(input_size, hidden_size).num_layers(num_layers));
      // fc = torch::nn::Linear(hidden_size, 6);

      lstm = register_module("lstm", lstm);
      fc = register_module("fc", fc);
    }

    torch::Tensor forward(torch::Tensor input) {
      auto lstm_out = lstm->forward(input);
      auto output = fc->forward(std::get<0>(lstm_out));
      return output;
    }
    torch::nn::Linear fc;
};

int main() {
  int64_t input_size = 6;
  int64_t hidden_size = 256;
  int64_t num_layers = 2;
  int64_t output_size = 6;

  LSTMImpl lstm(input_size, hidden_size, num_layers, output_size);
  torch::nn::init::constant_(lstm.fc->bias, 0.5);

  torch::optim::Adam optimizer(lstm.parameters(), torch::optim::AdamOptions(0.001));

  torch::Tensor input = torch::randn({10, 1, input_size});

  lstm.eval();
  torch::Tensor output = lstm.forward(input);

  std::cout << "Input:\n" << input << std::endl;

  std::cout << "Output:\n" << output << std::endl;
}
