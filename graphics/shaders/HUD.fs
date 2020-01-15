#version 330
in vec2 newTexture;
in vec3 fragNormal;

out vec4 outColor;
uniform sampler2D samplerTexture;

void main()
{
	vec4 texel = texture(samplerTexture, newTexture);
	outColor = texel;
}
