class_names=['0','1','2','3','4','5','6','7','8','9']
from sklearn.metrics import classification_report

def generate_report(Y_true,Y_predicted):
    print(classification_report(Y_true,Y_predicted,target_names=class_names,digits=4))