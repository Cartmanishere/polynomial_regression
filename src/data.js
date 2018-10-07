import * as tf from '@tensorflow/tfjs';

export function generateData(numPoints, coeff, sigma = 0.04) {
    return tf.tidy(() => {
        const [a, b, c] = [
            tf.scalar(coeff.a), tf.scalar(coeff.b), tf.scalar(coeff.c)
        ];

        const xs = tf.randomUniform([numPoints], -1, 1);

        // Generate polynomial data
        const two = tf.scalar(2)
        const ys = a.mul(xs.pow(two)).add(b.mul(xs)).add(c).add(tf.randomNormal([numPoints], 0, sigma));
        return {
            xs,
            ys: ys
        };
    })
}