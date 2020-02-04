#version 330
in vec2 newTexture;
in vec3 fragNormal;

out vec4 outColor;
uniform sampler2D samplerTexture;

void main()
{
	vec3 ambientLightIntensity = vec3(0.9f,0.9f,0.9f);
	vec3 sunLightIntensity = vec3(0.1f,0.1f,0.1f);
	vec3 sunLightDirection = normalize(vec3(0.0f,0.0f,0.01f));

	vec4 texel = texture(samplerTexture, newTexture);
	vec3 lightIntensity = ambientLightIntensity+sunLightIntensity*max(dot(fragNormal,sunLightDirection), 0.0f);
    outColor = vec4(texel.rgb*lightIntensity, texel.a);
}
