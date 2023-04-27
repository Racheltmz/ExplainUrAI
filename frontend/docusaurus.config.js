module.exports = {
    plugins: [
      [
        'docusaurus-plugin-react-docgen-typescript',
        {
          /** @type {import('docusaurus-plugin-react-docgen-typescript').Options} */
          // pass in a single string or an array of strings
          src: ['./src/**/*.tsx', '!./src/**/*test.*'],
          parserOptions: {
            // pass parserOptions to react-docgen-typescript
            // here is a good starting point which filters out all
            // types from react
            propFilter: (prop, component) => {
              if (prop.parent) {
                return !prop.parent.fileName.includes('@types/react');
              }
  
              return true;
            },
          },
        },
      ],
    ],
  };