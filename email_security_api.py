import pickle
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

class EmailSecurityAnalyzer:
    """Combined fraud and prompt injection detection"""
    
    def __init__(self):
        # Load fraud detection model
        with open('Model/fraud_detector.pkl', 'rb') as f:
            self.fraud_model = pickle.load(f)
        with open('Model/vectorizer.pkl', 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        # Load prompt injection model
        self.injection_pipe = pipeline(
            "text-classification",
            model="protectai/deberta-v3-base-prompt-injection-v2"
        )
    
    def analyze(self, email_text):
        """Analyze email for fraud and prompt injection"""
        
        # Fraud detection
        email_vec = self.vectorizer.transform([email_text])
        fraud_pred = self.fraud_model.predict(email_vec)[0]
        fraud_prob = self.fraud_model.predict_proba(email_vec)[0]
        
        # Prompt injection detection
        injection_result = self.injection_pipe(email_text[:512])[0]
        
        return {
            'is_safe': fraud_pred == 0 and injection_result['label'] != 'INJECTION',
            'fraud_detected': bool(fraud_pred),
            'fraud_confidence': float(fraud_prob[1]),
            'injection_detected': injection_result['label'] == 'INJECTION',
            'injection_confidence': float(injection_result['score']),
            'risk_level': self._calculate_risk(fraud_prob[1], injection_result)
        }
    
    def _calculate_risk(self, fraud_prob, injection_result):
        """Calculate overall risk level"""
        if fraud_prob > 0.8 or (injection_result['label'] == 'INJECTION' and injection_result['score'] > 0.8):
            return 'CRITICAL'
        elif fraud_prob > 0.5 or injection_result['label'] == 'INJECTION':
            return 'HIGH'
        elif fraud_prob > 0.3:
            return 'MEDIUM'
        else:
            return 'LOW'

# Example usage
if __name__ == "__main__":
    analyzer = EmailSecurityAnalyzer()
    
    email = "Urgent! Your account will be suspended. Click here to verify your password."
    result = analyzer.analyze(email)
    
    print("Analysis Result:")
    print(f"  Safe: {result['is_safe']}")
    print(f"  Fraud Detected: {result['fraud_detected']} ({result['fraud_confidence']:.2%})")
    print(f"  Injection Detected: {result['injection_detected']} ({result['injection_confidence']:.2%})")
    print(f"  Risk Level: {result['risk_level']}")
