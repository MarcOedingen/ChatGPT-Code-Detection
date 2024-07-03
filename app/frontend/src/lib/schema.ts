import { z } from 'zod';
import { codeExample } from './constants';

export const formSchema = z.object({
	code: z.string().default(codeExample)
});

export const classifySchema = z.object({
	prediction: z.number(),
	probability: z.number()
});

export type FormSchema = typeof formSchema;
export type ClassifySchema = typeof classifySchema;
