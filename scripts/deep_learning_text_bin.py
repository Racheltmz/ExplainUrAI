# Import classes
from deep_learning_text_base import DLText

# Import Libraries
import argparse

# Deep Learning Text Tasks (Binary)
class DLTextBinary(DLText):
    def __init__(self, model, text, classes, NUM_FEATURES=5, NUM_SAMPLES=1000, need_process=False, ds=None, ds_text_field=None, MAX_NUM=None, MAX_SEQ=None):
        super().__init__(model, text, classes, NUM_FEATURES, NUM_SAMPLES, need_process, ds, ds_text_field, MAX_NUM, MAX_SEQ)
    
    def generate_report(self):
        html_filename = f'{super().get_report_convention}dl_textbinary_{super().get_current_epoch}.html'
        # Get explanation
        exp = super().get_explanation(self.classes, self.need_process, task_type='binary')
        # Write to a html file
        with open(html_filename, 'w', encoding='utf-8') as f:
            f.write(exp.as_html())
        f.close()

if __name__ == "__main__":
    # Define arguments
    parser = argparse.ArgumentParser(description='Explain Predictions of Text Classification Model')
    parser.add_argument(
        '--filetype', default='pdf', choices=['pdf', 'html'], help='report file type', type=str
    )
    parser.add_argument(
        'model', default='', help='path to model file (h5 format)', type=str
    )
    parser.add_argument(
        'text', default='', help='text to classify', type=str
    )
    parser.add_argument(
        'classes', default='', help='list of classes', type=str, nargs='+'
    )
    parser.add_argument(
        '--num_features', default=5, help='maximum number of features present in explanation', type=int
    )
    parser.add_argument(
        '--num_samples', default=1000, help='size of the neighborhood to learn the linear model', type=int
    )
    parser.add_argument(
        '--tokenize', default=False, help='set to True if model does not have a tokenizer layer', type=bool
    )
    parser.add_argument(
        '--dataset', default=None, help='dataset file', type=str
    )
    parser.add_argument(
        '--label_field', default=None, help='label field name in dataset file', type=str
    )
    parser.add_argument(
        '--max_num', default=None, help='maximum number for tokenization', type=int
    )
    parser.add_argument(
        '--max_seq', default=None, help='maximum sequence for padding', type=int
    )
    args = parser.parse_args()

    # Generate report
    dl_textbin_xai = DLTextBinary(
        args.model, 
        args.text, 
        args.classes, 
        args.num_features, 
        args.num_samples, 
        args.tokenize, 
        args.dataset,
        args.label_field,
        args.max_num,
        args.max_seq)
    dl_textbin_xai.generate_report()
