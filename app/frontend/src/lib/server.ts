import { env } from '$env/dynamic/private';
import type { z } from 'zod';
import { classifySchema, type ClassifySchema } from './schema';

class ModelService {
	apiUrl: string;

	constructor(apiUrl: string) {
		if (!apiUrl) {
			throw new Error('API_URL is not defined');
		}

		this.apiUrl = apiUrl;
	}
	async classify(code: string): Promise<z.infer<ClassifySchema>> {
		try {
			const response = await fetch(`${this.apiUrl}/predictions`, {
				mode: 'no-cors',
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify({ input: { code: code } })
			});

			const { output } = await response.json();
			const result = classifySchema.parse(output);
			return result;
		} catch (error) {
			console.log(error);
			throw new Error('Something went wrong. Please try again later.');
		}
	}
}

const modelService = new ModelService(env.API_URL || 'http://localhost:5002');

export default modelService;
