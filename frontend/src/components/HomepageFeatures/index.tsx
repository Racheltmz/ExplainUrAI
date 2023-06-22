import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  Svg: React.ComponentType<React.ComponentProps<'svg'>>;
  description: JSX.Element;
};

const FeatureList: FeatureItem[] = [
  {
    // <a href="https://www.flaticon.com/free-icons/visual-basic" title="visual basic icons">Visual basic icons created by surang - Flaticon</a>
    title: 'Image Classification',
    Svg: require('@site/static/img/image_classification.svg').default,
    description: (
      <>
        For your Convolutional Neural Network models that you have trained
        for image classification
      </>
    ),
  },
  {
    // <a href="https://www.flaticon.com/free-icons/files-and-folders" title="files-and-folders icons">Files-and-folders icons created by itim2101 - Flaticon</a>
    title: 'Text Classification',
    Svg: require('@site/static/img/text_classification.svg').default,
    description: (
      <>
        Capable in handling a few Natural Language Processing tasks such as 
        Text Classification and Sentiment Analysis
      </>
    ),
  },
  {
    // <a href="https://www.flaticon.com/free-icons/logistic-regression" title="logistic regression icons">Logistic regression icons created by Freepik - Flaticon</a>
    title: 'Regression',
    Svg: require('@site/static/img/regression.svg').default,
    description: (
      <>
        For your linear regression models built with packages such as scikit-learn.
      </>
    ),
  },
  {
    // <a href="https://www.flaticon.com/free-icons/visual" title="visual icons">Visual icons created by Freepik - Flaticon</a>
    title: 'Classification',
    Svg: require('@site/static/img/classification.svg').default,
    description: (
      <>
        For your classification models built with packages such as scikit-learn.
      </>
    ),
  },
];

function Feature({title, Svg, description}: FeatureItem) {
  return (
    <div className={clsx('col col--6')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): JSX.Element {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
