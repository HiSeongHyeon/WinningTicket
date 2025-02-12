# utils.py
import os
import time
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 다중 GPU 사용 시 필요
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Model.py에 정의된 WDCNN 모델 임포트
from Model import WDCNN

#################################################
# 1. UOS 데이터셋 및 DataLoader 관련 함수들
#################################################
# UOS 데이터셋 기본 경로 (필요에 따라 수정)
base_path = r"C:\Users\ChoiSeongHyeon\Desktop\WinningT\Winning Ticket\MyWinningTicket\Dataset\UOS"
parameters_dir = r"C:\Users\ChoiSeongHyeon\Desktop\WinningT\Winning Ticket\MyWinningTicket\UOS\Parameters"
original_parameters_dir = os.path.join(parameters_dir, "Original")
unstructured_parameters_dir = os.path.join(parameters_dir, "Unstructured")
structured_parameters_dir = os.path.join(parameters_dir, "Structured")
os.makedirs(original_parameters_dir, exist_ok=True)
os.makedirs(unstructured_parameters_dir, exist_ok=True)
os.makedirs(structured_parameters_dir, exist_ok=True)
# Fault Type 라벨링 (H:0, IR:1, OR:2, B:3)
fault_types = {'H': 0, 'IR': 1, 'OR': 2, 'B': 3}

class UOSDataset(Dataset):
    """
    UOS 데이터셋 클래스  
    각 CSV 파일은 2,048 길이의 1차원 신호 데이터로 구성되며,  
    파일명에 "Train", "Validation", "Test" 문자열이 포함되어 데이터를 구분합니다.
    """
    def __init__(self, dataset_type: str):
        """
        Args:
            dataset_type (str): "Train", "Validation" 또는 "Test"
        """
        self.data = []
        self.labels = []
        for fault, label in fault_types.items():
            sample_path = os.path.join(base_path, fault, "Samples")
            if not os.path.exists(sample_path):
                print(f"Warning: Path does not exist - {sample_path}")
                continue
            for file in os.listdir(sample_path):
                if file.endswith('.csv') and dataset_type in file:
                    file_path = os.path.join(sample_path, file)
                    try:
                        # CSV 파일을 읽어 1차원 배열로 변환 (header 없음)
                        data = pd.read_csv(file_path, header=None).values.flatten()
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
                        continue
                    # 데이터 길이가 2,048인 경우에만 추가
                    if len(data) == 2048:
                        self.data.append(data)
                        self.labels.append(label)
        # 리스트를 텐서로 변환: shape -> (N, 1, 2048)
        self.data = torch.tensor(self.data, dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        # Validation 및 Test 데이터는 재현을 위해 한 번 섞음
        if dataset_type in ["Validation", "Test"]:
            import numpy as np
            np.random.seed(42)
            indices = np.random.permutation(len(self.data))
            self.data = self.data[indices]
            self.labels = self.labels[indices]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def get_dataloaders(batch_size=32):
    """
    Train, Validation, Test DataLoader를 반환하는 함수.
    """
    train_dataset = UOSDataset("Train")
    val_dataset = UOSDataset("Validation")
    test_dataset = UOSDataset("Test")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

#################################################
# 1.5. 기존 원본 모델 학습 및 평가 함수들
#################################################
def train_model_original(train_dataset, val_dataset, batch_sizes=[32, 64, 128, 256],
                         num_epochs=100, learning_rate=0.01, patience=10):
    """
    원본 모델을 학습하는 함수 (배치 크기를 변화시키며 학습하고, 최고 성능 모델을 저장합니다).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 저장 폴더: UOS\Parameters (환경에 맞게 경로 조정)
    best_overall_acc = 0.0
    best_batch_size = None
    best_model_path = os.path.join(parameters_dir, "best_overall_model.pth")
    best_initial_weights_path = os.path.join(parameters_dir, "best_initial_weights.pth")
    
    for batch_size in batch_sizes:
        print(f"\n🔹 Training with Batch Size: {batch_size}")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=lambda _: np.random.seed(42))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        model = WDCNN().to(device)
        init_weights_path = os.path.join(parameters_dir, f"initial_weights_bs{batch_size}.pth")
        torch.save(model.state_dict(), init_weights_path)
        print(f"✅ Initial weights saved for batch {batch_size}!")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        best_val_acc = 0.0
        early_stop_counter = 0
        for epoch in range(num_epochs):
            model.train()
            train_loss, train_correct = 0.0, 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            train_acc = train_correct / len(train_loader.dataset)
            model.eval()
            val_loss, val_correct = 0.0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    val_correct += (outputs.argmax(dim=1) == labels).sum().item()
            val_acc = val_correct / len(val_loader.dataset)
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}: Batch {batch_size}, LR: {current_lr:.6f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                early_stop_counter = 0
                best_model_bs_path = os.path.join(parameters_dir, f"best_model_bs{batch_size}.pth")
                torch.save(model.state_dict(), best_model_bs_path)
                print(f"✅ Best model saved (Batch {batch_size}) with val_acc {best_val_acc:.4f}")
            else:
                early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break
        if best_val_acc > best_overall_acc:
            best_overall_acc = best_val_acc
            best_batch_size = batch_size
            torch.save(model.state_dict(), best_model_path)
            best_initial_weights_path = init_weights_path
            print(f"\n🏆 New Best Overall Model Saved! Val Acc: {best_overall_acc:.4f} (Batch {best_batch_size})")
    print(f"\n✅ Training completed! Best model: {best_model_path} from batch size {best_batch_size}")
    print(f"🎯 Best Initial Weights Saved: {best_initial_weights_path}")

def evaluate_model_original(model, test_loader, device):
    """
    원본 모델을 테스트 데이터셋에서 평가하여 Classification Report 및 Confusion Matrix를 출력합니다.
    """
    model.to(device)
    model.eval()
    y_true = []
    y_pred = []
    from sklearn.metrics import classification_report, confusion_matrix
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())
    target_names = [k for k, v in sorted(fault_types.items(), key=lambda item: item[1])]
    print("\n📌 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))
    plt.figure(figsize=(6,5))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

#################################################
# 2. Unstructured Pruning 및 Fine-Tuning 함수들
#################################################
# Unstructured Pruning 저장 폴더 (예: UOS\Parameters\Unstructured)
PRUNING_SAVE_DIR = unstructured_parameters_dir
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def apply_sparse_training(model, amount):
    """
    모델의 모든 Conv1d 및 Linear 레이어에 대해 L1 기반 unstructured pruning 적용.
    """
    model = deepcopy(model)
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)
    return model

def remove_pruning_and_save(model, save_path):
    """
    Fine-Tuning 후 Pruning Mask 제거 후 모델 state_dict 저장.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
            prune.remove(module, "weight")
    torch.save(model.state_dict(), save_path)

def fine_tune_and_evaluate(pruning_amounts, train_loader, val_loader, num_epochs):
    """
    각 unstructured pruning 비율마다 미리 학습된 모델(best_overall_model.pth)을 불러와
    pruning을 적용한 후 Fine-Tuning을 진행하고 에포크별 Validation Accuracy를 기록한 후,
    최적 모델을 저장합니다.
    """
    results = []
    BEST_MODEL_PATH = os.path.join(os.path.dirname(PRUNING_SAVE_DIR), "best_overall_model.pth")
    for pruning_amount in pruning_amounts:
        print(f"\n🔹 Fine-tuning with Pruning Amount: {pruning_amount}")
        model = WDCNN().to(DEVICE)
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
        pruned_model = apply_sparse_training(model, amount=pruning_amount)
        pruned_model.to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(pruned_model.parameters(), lr=1e-5)
        best_val_acc = 0.0
        best_pruned_model = None
        val_accuracies = []
        for epoch in range(num_epochs):
            pruned_model.train()
            train_correct = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = pruned_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            train_acc = train_correct / len(train_loader.dataset)
            pruned_model.eval()
            val_correct = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    outputs = pruned_model(inputs)
                    val_correct += (outputs.argmax(dim=1) == labels).sum().item()
            val_acc = val_correct / len(val_loader.dataset)
            val_accuracies.append(val_acc)
            print(f"Epoch {epoch+1}: Pruning {pruning_amount}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_pruned_model = deepcopy(pruned_model)
        model_save_path = os.path.join(PRUNING_SAVE_DIR, f"best_unstructured_finetuned_{int(pruning_amount*100)}.pth")
        remove_pruning_and_save(best_pruned_model, model_save_path)
        results.append({
            "Pruning Ratio": pruning_amount,
            "Best Validation Accuracy": best_val_acc,
            "Validation Accuracy per Epoch": val_accuracies
        })
    return results

#################################################
# 3. 평가 및 분석 함수들 (공통)
#################################################
def count_total_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_nonzero_parameters(model):
    return sum((p != 0).sum().item() for p in model.parameters() if p.requires_grad)

def measure_inference_time(model, test_loader, device, num_samples=100):
    model.eval()
    total_time = 0.0
    num_runs = 0
    with torch.no_grad():
        for inputs, _ in test_loader:
            if num_runs >= num_samples:
                break
            inputs = inputs.to(device)
            start_time = time.time()
            _ = model(inputs)
            end_time = time.time()
            total_time += (end_time - start_time)
            num_runs += 1
    return total_time / num_runs if num_runs > 0 else 0.0

def evaluate_pruned_models(test_loader, experiment_results):
    """
    Test 데이터셋에서 원본 및 각 unstructured pruning 비율별 Fine-Tuning 결과의 성능을 평가합니다.
    테스트 정확도, 비-제로 파라미터 수, Inference Time을 계산하여 DataFrame으로 반환합니다.
    """
    test_accuracies = []
    BEST_MODEL_PATH = os.path.join(os.path.dirname(PRUNING_SAVE_DIR), "best_overall_model.pth")
    original_model = WDCNN().to(DEVICE)
    original_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    total_params = count_total_parameters(original_model)
    original_model.eval()
    test_correct = 0
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = original_model(inputs)
        test_correct += (outputs.argmax(dim=1) == labels).sum().item()
    original_test_acc = test_correct / len(test_loader.dataset)
    original_inference_time = measure_inference_time(original_model, test_loader, DEVICE)
    test_accuracies.append({
        "Pruning Ratio": 0.0,
        "Number of Non-Zero Params": total_params,
        "Inference Time (s)": original_inference_time,
        "Test Accuracy": original_test_acc
    })
    for result in experiment_results:
        pruning_ratio = result["Pruning Ratio"]
        model_path = os.path.join(PRUNING_SAVE_DIR, f"best_unstructured_finetuned_{int(pruning_ratio*100)}.pth")
        if not os.path.exists(model_path):
            print(f"❌ Model file not found for Pruning Ratio: {pruning_ratio}")
            continue
        model = WDCNN().to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        nonzero_params = count_nonzero_parameters(model)
        model.eval()
        test_correct = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                test_correct += (outputs.argmax(dim=1) == labels).sum().item()
        test_acc = test_correct / len(test_loader.dataset)
        inference_time = measure_inference_time(model, test_loader, DEVICE)
        test_accuracies.append({
            "Pruning Ratio": pruning_ratio,
            "Number of Non-Zero Params": nonzero_params,
            "Inference Time (s)": inference_time,
            "Test Accuracy": test_acc
        })
    return pd.DataFrame(test_accuracies)

def plot_validation_accuracies(original_val_accuracies, experiment_results, num_epochs):
    """
    원본 모델과 각 unstructured pruning 비율별 Fine-Tuning 결과의 Validation Accuracy 변화를 그래프로 그립니다.
    """
    plt.figure(figsize=(10,6))
    plt.plot(range(1, num_epochs+1), original_val_accuracies, label="Original Model (Pruning Ratio 0%)", linestyle="--", color="black")
    for result in experiment_results:
        pruning_ratio = result["Pruning Ratio"]
        val_accuracies = result["Validation Accuracy per Epoch"]
        plt.plot(range(1, len(val_accuracies)+1), val_accuracies, label=f"Pruning Ratio {pruning_ratio*100:.0f}%")
    plt.xlabel("Training Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy of WDCNN with Different Unstructured Pruning Ratios (UOS Dataset)")
    plt.legend()
    plt.grid(True)
    plt.show()

#################################################
# 4. Structured Pruning 및 Fine-Tuning 함수들
#################################################
# Structured Pruning 저장 폴더 (예: UOS\Parameters\Structured)
STRUCTURED_SAVE_DIR = structured_parameters_dir

def apply_structured_pruning(model, amount, n=2, dim=0):
    """
    모델의 모든 Conv1d 및 Linear 레이어에 대해 LN 기반 structured pruning을 적용합니다.
    
    Args:
        amount (float): pruning 비율
        n (int): L_n norm (보통 2)
        dim (int): pruning 할 차원 (예: conv의 경우 전체 필터(prune dim=0))
    """
    model = deepcopy(model)
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
            prune.ln_structured(module, name="weight", amount=amount, n=n, dim=dim)
    return model

def remove_structured_pruning_and_save(model, save_path):
    """
    Fine-Tuning 후 Structured Pruning Mask를 제거하고 모델 state_dict를 저장합니다.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
            prune.remove(module, "weight")
    torch.save(model.state_dict(), save_path)

def fine_tune_and_evaluate_structured(pruning_amounts, train_loader, val_loader, num_epochs, n=2, dim=0):
    """
    각 structured pruning 비율마다 미리 학습된 모델(best_overall_model.pth)을 불러와
    structured pruning을 적용한 후 Fine-Tuning을 진행하고 에포크별 Validation Accuracy를 기록하여,
    최적 모델을 저장합니다.
    """
    results = []
    BEST_MODEL_PATH = os.path.join(os.path.dirname(STRUCTURED_SAVE_DIR), "best_overall_model.pth")
    for pruning_amount in pruning_amounts:
        print(f"\n🔹 Structured Fine-tuning with Pruning Amount: {pruning_amount}")
        model = WDCNN().to(DEVICE)
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
        pruned_model = apply_structured_pruning(model, amount=pruning_amount, n=n, dim=dim)
        pruned_model.to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(pruned_model.parameters(), lr=1e-5)
        best_val_acc = 0.0
        best_pruned_model = None
        val_accuracies = []
        for epoch in range(num_epochs):
            pruned_model.train()
            train_correct = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = pruned_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            train_acc = train_correct / len(train_loader.dataset)
            pruned_model.eval()
            val_correct = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    outputs = pruned_model(inputs)
                    val_correct += (outputs.argmax(dim=1) == labels).sum().item()
            val_acc = val_correct / len(val_loader.dataset)
            val_accuracies.append(val_acc)
            print(f"Epoch {epoch+1}: Structured Pruning {pruning_amount}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_pruned_model = deepcopy(pruned_model)
        model_save_path = os.path.join(STRUCTURED_SAVE_DIR, f"best_structured_finetuned_{int(pruning_amount*100)}.pth")
        remove_structured_pruning_and_save(best_pruned_model, model_save_path)
        results.append({
            "Pruning Ratio": pruning_amount,
            "Best Validation Accuracy": best_val_acc,
            "Validation Accuracy per Epoch": val_accuracies
        })
    return results

def evaluate_structured_models(test_loader, experiment_results):
    """
    Test 데이터셋에서 원본 및 각 structured pruning 비율별 Fine-Tuning 결과의 성능을 평가합니다.
    테스트 정확도, 비-제로 파라미터 수, Inference Time을 계산하여 DataFrame으로 반환합니다.
    """
    test_accuracies = []
    BEST_MODEL_PATH = os.path.join(os.path.dirname(STRUCTURED_SAVE_DIR), "best_overall_model.pth")
    original_model = WDCNN().to(DEVICE)
    original_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    total_params = count_total_parameters(original_model)
    original_model.eval()
    test_correct = 0
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = original_model(inputs)
        test_correct += (outputs.argmax(dim=1) == labels).sum().item()
    original_test_acc = test_correct / len(test_loader.dataset)
    original_inference_time = measure_inference_time(original_model, test_loader, DEVICE)
    test_accuracies.append({
        "Pruning Ratio": 0.0,
        "Number of Non-Zero Params": total_params,
        "Inference Time (s)": original_inference_time,
        "Test Accuracy": original_test_acc
    })
    for result in experiment_results:
        pruning_ratio = result["Pruning Ratio"]
        model_path = os.path.join(STRUCTURED_SAVE_DIR, f"best_structured_finetuned_{int(pruning_ratio*100)}.pth")
        if not os.path.exists(model_path):
            print(f"❌ Model file not found for Structured Pruning Ratio: {pruning_ratio}")
            continue
        model = WDCNN().to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        nonzero_params = count_nonzero_parameters(model)
        model.eval()
        test_correct = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                test_correct += (outputs.argmax(dim=1) == labels).sum().item()
        test_acc = test_correct / len(test_loader.dataset)
        inference_time = measure_inference_time(model, test_loader, DEVICE)
        test_accuracies.append({
            "Pruning Ratio": pruning_ratio,
            "Number of Non-Zero Params": nonzero_params,
            "Inference Time (s)": inference_time,
            "Test Accuracy": test_acc
        })
    return pd.DataFrame(test_accuracies)