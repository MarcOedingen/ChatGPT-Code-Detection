import { fail, message, superValidate } from 'sveltekit-superforms';
import { formSchema } from '../lib/schema';
import { zod } from 'sveltekit-superforms/adapters';
import type { Actions } from '@sveltejs/kit';
import modelService from '$lib/server';

export const load = async () => {
	const form = await superValidate(zod(formSchema));
	return {
		form
	};
};

export const actions: Actions = {
	default: async (event) => {
		const form = await superValidate(event, zod(formSchema));
		try {
			if (!form.valid) {
				return fail(400, {
					form
				});
			}
			const data = await modelService.classify(form.data.code);
			return {
				form,
				data
			};
		} catch (error) {
			if (error instanceof Error) {
				return message(form, error.message, {
					status: 500
				});
			}
			throw error;
		}
	}
};
