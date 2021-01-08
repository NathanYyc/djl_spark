package fdu.djl;

import ai.djl.modality.Classifications;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import org.apache.spark.ml.classification.Classifier;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class MyTranslator implements Translator<NDList, Classifications> {
    /**
     * Gets the {@link Batchifier}.
     *
     * @return the {@link Batchifier}
     */
    @Override
    public Batchifier getBatchifier() {
        return null;
    }

    /**
     * Processes the output NDList to the corresponding output object.
     *
     * @param ctx  the toolkit used for post-processing
     * @param list the output NDList after inference
     * @return the output object of expected type
     * @throws Exception if an error occurs during processing output
     */
    @Override
    public Classifications processOutput(TranslatorContext ctx, NDList list) throws Exception {
        NDArray probabilities = list.singletonOrThrow().softmax(0);
        List<String> classNames = IntStream.range(0, 10).mapToObj(String::valueOf).collect(Collectors.toList());
        return new Classifications(classNames,probabilities);
    }

    /**
     * Processes the input and converts it to NDList.
     *
     * @param ctx   the toolkit for creating the input NDArray
     * @param input the input object
     * @return the {@link NDList} after pre-processing
     * @throws Exception if an error occurs during processing input
     */
    @Override
    public NDList processInput(TranslatorContext ctx, NDList input) throws Exception {
        return input;
    }
}
