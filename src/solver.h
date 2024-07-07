#pragma once
#include "config.h"
#include "dataloader.h"
#include "loss.h"
#include "lr_scheduler.h"
#include "model.h"
#include "optimizer.h"

namespace micro_nn::solver {
template <class ModelT, class OptimizerT, class LossT, class DataLoaderT,
          class NumT,
          class LRSchedulerT = lr_scheduler::ConstantRateScheduler<NumT>>
    requires model::Model<NumT, ModelT> && optimizer::Optimizer<OptimizerT> &&
             loss::Loss<NumT, LossT> && data::DataLoader<NumT, DataLoaderT> &&
             lr_scheduler::LRScheduler<NumT, LRSchedulerT>
class Solver {
public:
    constexpr explicit Solver(ModelT& model, OptimizerT& optimizer, LossT& loss,
                              DataLoaderT& dataloader,
                              LRSchedulerT lr_scheduler = {})
        : model_{model},
          optimizer_{optimizer},
          loss_{loss},
          dataloader_{dataloader},
          lr_scheduler_{std::move(lr_scheduler)} {}

    constexpr void train(size_t epochs) {
        for (size_t epoch{0}; epoch < epochs; ++epoch) {
            while (dataloader_.has_next()) {
                auto batch{dataloader_.next()};
                auto x{batch.x};
                auto y_true{batch.y};
                step(x, y_true);
            }
            dataloader_.reset();
        }
    }

private:
    constexpr void step(const linalg::Matrix<NumT>& x,
                        const linalg::Matrix<NumT>& y_true) {
        auto y_pred{model_.forward(x)};
        const auto loss{loss_.forward(y_true, y_pred)};
        const auto grad{loss_.backward(y_true, y_pred)};
        model_.backward(grad);
        optimizer_.step();
    }

    ModelT& model_;
    OptimizerT& optimizer_;
    LossT& loss_;
    DataLoaderT& dataloader_;
    LRSchedulerT lr_scheduler_;
};

template <class ModelT, class OptimizerT, class LossT, class DataLoaderT,
          class LRSchedulerT>
Solver(ModelT, OptimizerT, LossT, DataLoaderT,
       LRSchedulerT) -> Solver<ModelT, OptimizerT, LossT, DataLoaderT,
                               typename DataLoaderT::ElementT, LRSchedulerT>;

template <class ModelT, class OptimizerT, class LossT, class DataLoaderT>
Solver(ModelT, OptimizerT, LossT, DataLoaderT)
    -> Solver<
        ModelT, OptimizerT, LossT, DataLoaderT, typename DataLoaderT::ElementT,
        lr_scheduler::ConstantRateScheduler<typename DataLoaderT::ElementT>>;

}  // namespace micro_nn::solver