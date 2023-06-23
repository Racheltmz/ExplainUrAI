# Import classes
from deep_learning_text_base import DLText

# Import Libraries
import argparse
from bs4 import BeautifulSoup

# Deep Learning Text Tasks (Multi-class or Multi-label)
class DLTextMulti(DLText):
    def __init__(self, model, text, classes, NUM_FEATURES=5, NUM_SAMPLES=1000, need_process=False, ds=None, ds_text_field=None, MAX_NUM=None, MAX_SEQ=None):
        super().__init__(model, text, classes, NUM_FEATURES, NUM_SAMPLES, need_process, ds, ds_text_field, MAX_NUM, MAX_SEQ)

    # Add multiple plotly graphs to a html file
    def figures_to_html(self, figs, filename='report.html'):
        with open(filename, 'w', encoding='utf-8') as dashboard:
            dashboard.write('<html><head></head><body>' + '\n')
            for fig in figs:
                inner_html = fig.to_html().split('<body>')[1].split('</body>')[0]
                dashboard.write(inner_html)
            dashboard.write('</body></html>' + '\n')

    # Add multiple explanation visualisations to a html file
    def explanations_to_html(self, exps, filename='report.html'):
        # Iterate through each explanation
        for (idx, exp) in enumerate(exps):
            # If it is the first record, parse the html page with BeautifulSoup to get the body tag
            if idx == 0:
                # Get explanation as a html file
                html = exp.as_html()
                soup = BeautifulSoup(html, 'html.parser')
                old_body = soup.find('body')
            else:
                html_to_extract = exp.as_html()
                soup_to_extract = BeautifulSoup(html_to_extract, 'html.parser')
                visualisation_tags = soup_to_extract.find('body').find_all(recursive=False)
                for tag in visualisation_tags:
                    old_body.append(tag)

        with open(filename, 'w', encoding='utf-8') as report:
            report.write(soup.prettify())
        report.close()

    # Generate report
    def generate_report(self):
        html_filename = f'{super().get_report_convention}dl_textmulti_{super().get_current_epoch}.html'
        # Instantiate list to store all explanations
        exps = []
        # Get the prediction for each label
        for label in self.classes:
            # Get the explanation and append it to a list
            exp = super().get_explanation(self.classes, self.need_process, 'multi', label)
            exps.append(exp)
        self.explanations_to_html(exps, html_filename)

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
    dl_textbin_xai = DLTextMulti(
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
